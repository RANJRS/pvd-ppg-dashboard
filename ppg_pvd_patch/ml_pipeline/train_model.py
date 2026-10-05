"""
train_model.py

Trains a small MLP (8 -> 16 -> 8 -> 3) on the extracted feature vectors to
classify PVD risk: Normal / Moderate / High. Kept intentionally small so the
quantized TFLite Micro model fits comfortably on the ESP32-S3 with room for
the DSP buffers.

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
]


def build_model(input_dim=8, num_classes=3):
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
    parser.add_argument("--data", default="./data/features.csv")
    parser.add_argument("--out", default="./model")
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

    # Subject-wise split: 80/20, no patient appears in both sets.
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
    np.save(os.path.join(args.out, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(args.out, "scaler_scale.npy"), scaler.scale_)

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
    main()
