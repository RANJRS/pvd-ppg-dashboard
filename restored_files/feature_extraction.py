

















































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


from scipy.stats import skew, kurtosis

def heart_rate(peaks):
    if len(peaks) < 2:
        return 0.0
    ts = [p["timestamp_ms"] for p in peaks]
    ibi_ms = np.mean(np.diff(ts))
    return float(60000.0 / ibi_ms) if ibi_ms > 0 else 0.0

def crest_factor(signal):
    rms = np.sqrt(np.mean(np.square(signal)))
    return float(np.max(np.abs(signal)) / rms) if rms > 1e-6 else 0.0

def signal_skewness(signal):
    return float(skew(signal))

def signal_kurtosis(signal):
    return float(kurtosis(signal))


def extract_features_row(row):
    finger_peaks = detect_peaks(row["filtered_finger"], row["timestamps_ms"])

    dc_finger = np.mean(row["raw_finger"])
    signal = np.array(row["filtered_finger"])

    return [
        perfusion_index(finger_peaks, dc_finger),
        augmentation_index(finger_peaks),
        hrv_rmssd(finger_peaks),
        dicrotic_ratio(finger_peaks),
        heart_rate(finger_peaks),
        crest_factor(signal),
        signal_skewness(signal),
        signal_kurtosis(signal),
    ]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(script_dir, "data", "processed.json"))
    parser.add_argument("--out", default=os.path.join(script_dir, "data", "features.csv"))
    args = parser.parse_args()

    df = pd.read_json(args.data)
    feature_cols = [
        "PI", "AI", "HRV_rmssd", "dicrotic_ratio",
        "heart_rate", "crest_factor", "skewness", "kurtosis"
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
