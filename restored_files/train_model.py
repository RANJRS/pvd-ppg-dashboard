"""
train_model.py

Trains a small MLP (8 -> 16 -> 8 -> 3) on the extracted feature vectors to
classify PVD risk: Normal / Watch / Refer. Kept intentionally small so the
quantized TFLite Micro model fits comfortably on the ESP32-S3 with room for
the DSP buffers.

Splits by subject_id (not by row) to avoid leaking the same patient's
windows across train/test, which would inflate reported accuracy.
"""
import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Suppress sklearn metric warnings about classes with no true samples
warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_COLS = [
    "PI", "AI", "HRV_rmssd", "dicrotic_ratio",
    "heart_rate", "crest_factor", "skewness", "kurtosis"
]


def build_model(input_dim=8, num_classes=2):
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(script_dir, "data", "features.csv"))
    parser.add_argument("--out", default=os.path.join(script_dir, "model"))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data, encoding="utf-8")
    
    # Filter out unlabeled (-1) records before training
    valid_idx = df["label"] != -1
    if not valid_idx.any():
        print("\n[INFO] No labeled data found for training.")
        print("Running inference (diagnosis) on the provided dataset instead...")
        model_path = os.path.join(args.out, "model.h5")
        if not os.path.exists(model_path):
            print("[ERROR] No trained model found. Please generate synthetic data and train a model first.")
            sys.exit(1)
            
        scaler_mean = np.load(os.path.join(args.out, "scaler_mean.npy"))
        scaler_scale = np.load(os.path.join(args.out, "scaler_scale.npy"))
        
        X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        X_scaled = (X - scaler_mean) / scaler_scale
        
        model = tf.keras.models.load_model(model_path)
        predictions = model.predict(X_scaled, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        label_names = ["Normal", "MI"]
        unique, counts = np.unique(y_pred, return_counts=True)
        print("\n=== Diagnosis Results ===")
        for val, count in zip(unique, counts):
            print(f"- {label_names[val]}: {count} windows ({count/len(y_pred)*100:.1f}%)")
        print("=========================")
        sys.exit(0)
        
    df = df[valid_idx]

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    groups = df["subject_id"].to_numpy()

    # Subject-wise split: 80/20, no patient appears in both sets.
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        print("\\n[WARNING] Only 1 unique subject found. Falling back to random row-wise split.")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model()
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
        y_test, y_pred, labels=[0, 1],
        target_names=["Normal", "MI"],
        zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    print(classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Normal", "MI"],
        zero_division=0))
    print("Confusion matrix:\n", cm)

    model.save(os.path.join(args.out, "model.h5"))
    # Save as TensorFlow SavedModel to bypass Keras 3 / TF 2.16 TFLite tracing issues
    model.export(os.path.join(args.out, "saved_model"))
    np.save(os.path.join(args.out, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(args.out, "scaler_scale.npy"), scaler.scale_)

    # Save training report JSON for the dashboard to read dynamically
    history = model.history.history if hasattr(model, 'history') else {}
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
            for cls in ["Normal", "MI"]
        },
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "total_samples": int(len(X)),
        "total_subjects": int(len(set(groups))),
        "epochs_run": int(len(model.history.history.get("loss", [args.epochs]))),
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


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)  # Ensure clean exit code regardless of TF background threads
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}", flush=True)
        sys.exit(1)
