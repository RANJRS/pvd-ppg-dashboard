"""
data_preprocessing.py

Loads raw dual-site PPG recordings (finger + toe), applies the same
bandpass filter used on-device, and segments into fixed-length windows
ready for feature extraction.

Expected raw data layout (one CSV per subject/session):
    ./data/raw/<subject_id>_<label>.csv
    columns: timestamp_ms, ir_finger, ir_toe
    label in filename: 'normal' | 'moderate' | 'high'  (clinician-assigned,
    e.g. from ABI test results: ABI>0.9 normal, 0.7-0.9 moderate, <0.7 high)

Output: a single processed.csv with one row per detected pulse-window,
columns = raw signal segments + timestamps + label, ready for
feature_extraction.py.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

FS = 200  # Hz, must match firmware config.h PPG_SAMPLE_RATE_HZ
BANDPASS = (0.5, 8.0)
WINDOW_SECONDS = 8
WINDOW_LEN = FS * WINDOW_SECONDS
STEP_SECONDS = 4  # 50% overlap between windows

LABEL_MAP = {"normal": 0, "moderate": 1, "high": 2}


def design_bandpass():
    """Regenerate this and re-derive the biquad coefficients in
    firmware/src/signal_processing.cpp if FS or BANDPASS changes."""
    sos = butter(2, BANDPASS, btype="bandpass", fs=FS, output="sos")
    return sos


def filter_signal(x, sos):
    return sosfiltfilt(sos, x)


def segment_windows(df, sos, label, subject_id):
    rows = []
    step = STEP_SECONDS * FS
    n = len(df)
    for start in range(0, n - WINDOW_LEN, step):
        seg = df.iloc[start:start + WINDOW_LEN]
        ir_finger = seg["ir_finger"].to_numpy(dtype=float)
        ir_toe = seg["ir_toe"].to_numpy(dtype=float)
        ts = seg["timestamp_ms"].to_numpy(dtype=float)

        filt_finger = filter_signal(ir_finger, sos)
        filt_toe = filter_signal(ir_toe, sos)

        rows.append({
            "subject_id": subject_id,
            "window_start_ms": ts[0],
            "raw_finger": ir_finger.tolist(),
            "raw_toe": ir_toe.tolist(),
            "filtered_finger": filt_finger.tolist(),
            "filtered_toe": filt_toe.tolist(),
            "timestamps_ms": ts.tolist(),
            "label": label,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="./data/raw")
    parser.add_argument("--out", default="./data/processed.csv")
    args = parser.parse_args()

    sos = design_bandpass()
    all_rows = []

    files = glob.glob(os.path.join(args.raw_dir, "*.csv"))
    if not files:
        print(f"No raw files found in {args.raw_dir}. "
              f"Expected files named <subject_id>_<label>.csv")
        return

    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        parts = base.split("_")
        label_str = parts[-1].lower()
        subject_id = "_".join(parts[:-1])

        if label_str not in LABEL_MAP:
            print(f"Skipping {f}: unrecognized label '{label_str}'")
            continue

        df = pd.read_csv(f)
        required_cols = {"timestamp_ms", "ir_finger", "ir_toe"}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {f}: missing columns {required_cols - set(df.columns)}")
            continue

        rows = segment_windows(df, sos, LABEL_MAP[label_str], subject_id)
        all_rows.extend(rows)
        print(f"{f}: {len(rows)} windows extracted")

    out_df = pd.DataFrame(all_rows)
    out_df.to_json(args.out.replace(".csv", ".json"), orient="records")
    print(f"Wrote {len(out_df)} windows to {args.out.replace('.csv', '.json')}")
    print("Run feature_extraction.py next to compute the 8-feature vectors.")


if __name__ == "__main__":
    main()