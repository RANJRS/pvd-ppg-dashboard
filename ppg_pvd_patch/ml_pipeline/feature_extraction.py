"""
feature_extraction.py

Computes the same 8-feature vector as firmware/src/feature_extraction.cpp,
from the windowed output of data_preprocessing.py. Keep the two
implementations in lockstep -- this is what lets a Python-trained model
run correctly on-device without a train/inference mismatch.

Feature order:
  0 PI_finger        6 HRV_rmssd
  1 PI_toe           7 dicrotic_ratio (toe)
  2 PI_ratio (toe/finger)
  3 PTT_ft (ms)
  4 AI_finger
  5 AI_toe
"""
import argparse
import numpy as np
import pandas as pd

FS = 200
MIN_PEAK_DISTANCE_MS = 300
MIN_PEAK_DISTANCE_SAMPLES = int(MIN_PEAK_DISTANCE_MS * FS / 1000)


def detect_peaks(filtered, timestamps_ms):
    """Adaptive-threshold local-max peak detector with dicrotic notch
    search, mirroring firmware/src/signal_processing.cpp::detectPeaks."""
    filtered = np.asarray(filtered)
    timestamps_ms = np.asarray(timestamps_ms)
    n = len(filtered)
    if n < 3:
        return []

    mean = filtered.mean()
    threshold = mean + 0.5 * filtered.std()

    peaks = []
    last_peak_idx = -MIN_PEAK_DISTANCE_SAMPLES
    for i in range(1, n - 1):
        is_local_max = filtered[i] > filtered[i - 1] and filtered[i] >= filtered[i + 1]
        if is_local_max and filtered[i] > threshold:
            if (i - last_peak_idx) < MIN_PEAK_DISTANCE_SAMPLES:
                continue

            search_start = i + int(150 * FS / 1000)
            search_end = min(n, i + int(400 * FS / 1000))
            dicrotic_amp = 0.0
            for j in range(search_start, search_end - 1):
                if j <= 0:
                    continue
                if filtered[j] > filtered[j - 1] and filtered[j] >= filtered[j + 1]:
                    dicrotic_amp = filtered[j]
                    break

            peaks.append({
                "index": i,
                "timestamp_ms": timestamps_ms[i],
                "amplitude": filtered[i],
                "dicrotic_notch_amplitude": dicrotic_amp,
            })
            last_peak_idx = i
    return peaks


def perfusion_index(peaks, dc_mean):
    if not peaks or dc_mean <= 0:
        return 0.0
    ac_mean = np.mean([p["amplitude"] for p in peaks])
    return (ac_mean / dc_mean) * 100.0


def pulse_transit_time(proximal_peaks, distal_peaks):
    if not proximal_peaks or not distal_peaks:
        return 0.0
    deltas = []
    for dp in distal_peaks:
        best_delta = None
        for pp in proximal_peaks:
            if dp["timestamp_ms"] <= pp["timestamp_ms"]:
                continue
            delta = dp["timestamp_ms"] - pp["timestamp_ms"]
            if delta < 50 or delta > 400:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
        if best_delta is not None:
            deltas.append(best_delta)
    return float(np.mean(deltas)) if deltas else 0.0


def augmentation_index(peaks):
    vals = [
        (p["dicrotic_notch_amplitude"] / p["amplitude"]) * 100.0
        for p in peaks if p["amplitude"] > 0
    ]
    return float(np.mean(vals)) if vals else 0.0


def hrv_rmssd(peaks):
    if len(peaks) < 3:
        return 0.0
    ts = [p["timestamp_ms"] for p in peaks]
    ibi = np.diff(ts)
    diffs = np.diff(ibi)
    return float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0


def dicrotic_ratio(peaks):
    vals = [
        p["dicrotic_notch_amplitude"] / p["amplitude"]
        for p in peaks if p["amplitude"] > 0
    ]
    return float(np.mean(vals)) if vals else 0.0


def extract_features_row(row):
    finger_peaks = detect_peaks(row["filtered_finger"], row["timestamps_ms"])
    toe_peaks = detect_peaks(row["filtered_toe"], row["timestamps_ms"])

    dc_finger = np.mean(row["raw_finger"])
    dc_toe = np.mean(row["raw_toe"])

    pi_finger = perfusion_index(finger_peaks, dc_finger)
    pi_toe = perfusion_index(toe_peaks, dc_toe)
    pi_ratio = (pi_toe / pi_finger) if pi_finger > 1e-3 else 0.0

    return [
        pi_finger,
        pi_toe,
        pi_ratio,
        pulse_transit_time(finger_peaks, toe_peaks),
        augmentation_index(finger_peaks),
        augmentation_index(toe_peaks),
        hrv_rmssd(finger_peaks),
        dicrotic_ratio(toe_peaks),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/processed.json")
    parser.add_argument("--out", default="./data/features.csv")
    args = parser.parse_args()

    df = pd.read_json(args.data)
    feature_cols = [
        "PI_finger", "PI_toe", "PI_ratio", "PTT_ft",
        "AI_finger", "AI_toe", "HRV_rmssd", "dicrotic_ratio",
    ]

    feature_rows = []
    for _, row in df.iterrows():
        feats = extract_features_row(row)
        feature_rows.append(feats + [row["label"], row["subject_id"]])

    out_df = pd.DataFrame(feature_rows, columns=feature_cols + ["label", "subject_id"])
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} feature rows to {args.out}")
    print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()
