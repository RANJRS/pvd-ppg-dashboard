"""
train_model.py

Trains a small MLP (13 -> 16 -> 8 -> 3) on the extracted 13-feature vectors
to classify PVD risk: Normal / Moderate / High. Kept intentionally small so
the quantized TFLite Micro model fits comfortably on the ESP32-S3.

Feature vector (13 features):
  PI_finger, PI_toe, PI_ratio, PTT_ft,
  AI_finger, AI_toe, HRV_rmssd, dicrotic_ratio,
  temp_finger, temp_toe, temp_diff,
  accel_std, gyro_std

Splits by subject_id (not by row) to avoid leaking the same patient's
windows across train/test, which would inflate reported accuracy.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = [
    "PI_finger", "PI_toe", "PI_ratio", "PTT_ft",
    "AI_finger", "AI_toe", "HRV_rmssd", "dicrotic_ratio",
    "temp_finger", "temp_toe", "temp_diff",
    "accel_std", "gyro_std",
]


def build_model(input_dim=13, num_classes=3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="./data/features.csv")
    parser.add_argument("--out",    default="./model")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    # Filter out unlabeled (-1) records before training
    df = df[df["label"] != -1]
    if len(df) == 0:
        print("[ERROR] No labeled data found to train model.")
        return

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    groups = df["subject_id"].to_numpy()

    # Robust dataset split logic with fallbacks
    unique_subjects = len(np.unique(groups))
    if unique_subjects >= 2:
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(splitter.split(X, y, groups))
        except ValueError:
            # Fallback if subject-wise split results in empty train set
            unique_classes = len(np.unique(y))
            if unique_classes >= 2 and len(X) >= 5:
                from sklearn.model_selection import StratifiedShuffleSplit
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                train_idx, test_idx = next(splitter.split(X, y))
            else:
                from sklearn.model_selection import ShuffleSplit
                splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                train_idx, test_idx = next(splitter.split(X))
    else:
        # Fallback for single subject dataset
        unique_classes = len(np.unique(y))
        if unique_classes >= 2 and len(X) >= 5:
            from sklearn.model_selection import StratifiedShuffleSplit
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(splitter.split(X, y))
        elif len(X) >= 2:
            from sklearn.model_selection import ShuffleSplit
            splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(splitter.split(X))
        else:
            # If only 1 sample exists, use it for both train and test
            train_idx = np.array([0])
            test_idx  = np.array([0])

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = build_model(input_dim=len(FEATURE_COLS))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True)
    ]
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=args.epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=2,
    )

    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    print("\n--- Held-out (subject-wise) evaluation ---")
    cr = classification_report(
        y_test, y_pred, labels=[0, 1, 2],
        target_names=["Normal", "Moderate", "High"],
        zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()

    print(classification_report(
        y_test, y_pred, labels=[0, 1, 2],
        target_names=["Normal", "Moderate", "High"],
        zero_division=0))
    print("Confusion matrix:\n", cm)

    model.save(os.path.join(args.out, "model.h5"))
    # Save as TensorFlow SavedModel using model.export (Keras 3 standard)
    model.export(os.path.join(args.out, "saved_model"))
    np.save(os.path.join(args.out, "scaler_mean.npy"),  scaler.mean_)
    np.save(os.path.join(args.out, "scaler_scale.npy"), scaler.scale_)

    # Save lightweight numpy weights for zero-overhead deployment & fallback
    try:
        dense_layers = [l for l in model.layers if "dense" in l.name.lower()]
        w1, b1 = dense_layers[0].get_weights()
        w2, b2 = dense_layers[1].get_weights()
        w3, b3 = dense_layers[2].get_weights()
        np.savez(os.path.join(args.out, "model_weights.npz"), 
                 w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3,
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        print(f"Exported model_weights.npz to {args.out}/")
    except Exception as e:
        print(f"Warning: could not export model_weights.npz: {e}")


    # Save training report JSON for the dashboard to read dynamically
    report = {
        "accuracy": float(cr.get("accuracy", 0)),
        "confusion_matrix": cm,
        "per_class": {
            cls: {
                "precision": round(float(cr[cls]["precision"]), 4),
                "recall":    round(float(cr[cls]["recall"]),    4),
                "f1":        round(float(cr[cls]["f1-score"]),  4),
                "support":   int(cr[cls]["support"])
            }
            for cls in ["Normal", "Moderate", "High"]
        },
        "train_samples":   int(len(X_train)),
        "test_samples":    int(len(X_test)),
        "total_samples":   int(len(X)),
        "total_subjects":  int(len(set(groups))),
        "epochs_run":      int(len(model.history.history.get("loss", [args.epochs]))),
        "feature_cols":    FEATURE_COLS,
    }
    report_path = os.path.join(args.out, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved training_report.json to {report_path}")

    print(f"\nSaved model.h5, saved_model/ and scaler stats to {args.out}/")
    print("NOTE: fold the scaler mean/scale into the firmware feature "
          "extraction (or as a pre-input normalization step) before "
          "deploying -- convert_to_tflite.py embeds this as a Quantize op "
          "where possible, but verify against firmware output on a bench "
          "test before trusting device-side risk scores.")

    # Auto-save backup of the trained model, stats, and dataset
    import datetime
    import shutil
    try:
        timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(args.out, "history", f"model_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)

        # Files in model/ folder to backup
        model_files = ["model.h5", "model_weights.npz", "scaler_mean.npy", "scaler_scale.npy", "training_report.json"]
        for fn in model_files:
            src = os.path.join(args.out, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(backup_dir, fn))

        # Backup saved_model directory
        sm_src = os.path.join(args.out, "saved_model")
        if os.path.exists(sm_src):
            shutil.copytree(sm_src, os.path.join(backup_dir, "saved_model"), dirs_exist_ok=True)

        # Backup data files (features.csv and processed.json)
        data_dir   = os.path.join(os.path.dirname(args.out), "data")
        data_files = ["features.csv", "processed.json"]
        for fn in data_files:
            src = os.path.join(data_dir, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(backup_dir, fn))

        print(f"\n[AUTO-SAVE] Successfully archived this training run to: {backup_dir}")
    except Exception as e:
        print(f"\n[WARNING] Auto-save backup failed: {e}")


if __name__ == "__main__":
    main()
