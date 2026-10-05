# ML Pipeline — PVD Risk Classifier

## Pipeline order

```
data_preprocessing.py  -> data/processed.json   (filtered, windowed dual-site PPG)
feature_extraction.py  -> data/features.csv      (8-feature vectors + labels)
train_model.py          -> model/model.h5, scaler_mean.npy, scaler_scale.npy
convert_to_tflite.py    -> ../firmware/model/model_data.h
```

## Data collection guidance (for capstone data logging)

Log raw finger + toe IR values straight off the firmware's `PpgSample`
struct over BLE or serial, timestamped, into CSVs named
`<subject_id>_<label>.csv` with columns `timestamp_ms, ir_finger, ir_toe`.

Label assignment should come from a reference measurement, not the PPG
device itself — e.g. Ankle-Brachial Index (ABI) via handheld Doppler:
- ABI > 0.9 → `normal`
- ABI 0.7–0.9 → `moderate`
- ABI < 0.7 → `high`

Even a modest cohort (n=20-30 diabetic + n=20-30 age-matched control) is
enough to get a defensible pilot classifier and preliminary
sensitivity/specificity numbers for a Scopus submission — just report them
as pilot/preliminary given the sample size.

## Class imbalance

Diabetic PVD screening cohorts are usually imbalanced toward `normal`.
`train_model.py` currently doesn't rebalance — if your collected data skews
heavily, add `class_weight` to `model.fit()` or oversample the minority
classes before training.

## Re-deploying after retraining

1. Re-run all four scripts in order.
2. Copy the printed scaler mean/scale into `feature_extraction.cpp` (or
   wire them in as `config.h` constants) so on-device normalization matches
   training-time normalization exactly — this is the most common source of
   train/inference mismatch in TinyML deployments.
3. Rebuild and reflash the firmware.
