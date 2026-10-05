import os
import numpy as np
import pandas as pd
import random

# Parameters matching firmware / data_preprocessing config
FS = 200  # Hz

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(script_dir, "data", "raw")
os.makedirs(raw_dir, exist_ok=True)

# Clear existing raw CSVs to start fresh
for f in os.listdir(raw_dir):
    if f.endswith('.csv'):
        try:
            os.remove(os.path.join(raw_dir, f))
        except Exception:
            pass

# Generate a random number of subjects (e.g. between 20 and 30)
num_subjects = random.randint(20, 30)
print(f"Generating random dataset with {num_subjects} subjects...")

labels = ["normal", "moderate", "high"]

for i in range(1, num_subjects + 1):
    subj_id = f"subj{i:02d}"
    label = random.choice(labels)

    # Randomize heart rate between 60 bpm (1.0 Hz) and 90 bpm (1.5 Hz)
    heart_rate = random.uniform(1.0, 1.5)

    # Randomize recording duration between 30 and 45 seconds
    duration = random.randint(30, 45)
    n_samples = FS * duration
    T = np.arange(n_samples) / FS
    ts = T * 1000.0  # ms

    # Set amplitude and delay parameters based on label with some random variations
    if label == "normal":
        A_toe = random.uniform(110.0, 140.0)
        ptt = random.uniform(0.10, 0.15)
        # Temperature: near-normal gradient (finger warmer by ~0.5°C)
        temp_finger_mean = random.uniform(34.0, 35.0)
        temp_toe_mean    = random.uniform(33.5, 34.5)
    elif label == "moderate":
        A_toe = random.uniform(65.0, 90.0)
        ptt = random.uniform(0.18, 0.23)
        # Temperature: moderate gradient (finger warmer by ~2-4°C)
        temp_finger_mean = random.uniform(33.5, 34.5)
        temp_toe_mean    = random.uniform(30.0, 32.5)
    else:  # high
        A_toe = random.uniform(30.0, 50.0)
        ptt = random.uniform(0.25, 0.32)
        # Temperature: large gradient (finger warmer by ~5-8°C) — ischaemic toes
        temp_finger_mean = random.uniform(33.0, 34.0)
        temp_toe_mean    = random.uniform(25.0, 28.5)

    # Finger signal (systolic peak + dicrotic notch secondary peak)
    ir_finger = (1000.0 + 200.0 * np.sin(2 * np.pi * heart_rate * T) +
                 60.0 * np.sin(4 * np.pi * heart_rate * T + 0.8) +
                 np.random.normal(0, 5.0, n_samples))

    # Toe signal: shifted by PTT and has different amplitude based on PVD severity
    ir_toe = (800.0 + A_toe * np.sin(2 * np.pi * heart_rate * (T - ptt)) +
              (A_toe * 0.3) * np.sin(4 * np.pi * heart_rate * (T - ptt) + 0.8) +
              np.random.normal(0, 3.0, n_samples))

    # --- Skin Temperature ---
    # Slow drift + small noise (thermistor bandwidth << PPG bandwidth)
    drift = 0.1 * np.sin(2 * np.pi * T / 60.0)   # very slow breathing/vasomotion
    temp_finger = temp_finger_mean + drift + np.random.normal(0, 0.05, n_samples)
    temp_toe    = temp_toe_mean    + drift + np.random.normal(0, 0.05, n_samples)

    # --- IMU (6-DOF, resting with mild motion artefact) ---
    # Acceleration in m/s²: gravity dominant on Z, small perturbations on X/Y
    ax = np.random.normal(0.0,  0.05, n_samples)   # lateral
    ay = np.random.normal(0.0,  0.05, n_samples)   # fore-aft
    az = np.random.normal(9.81, 0.05, n_samples)   # vertical (gravity)
    # Gyroscope in °/s: near-zero for resting patient, small noise
    gx = np.random.normal(0.0, 0.5, n_samples)
    gy = np.random.normal(0.0, 0.5, n_samples)
    gz = np.random.normal(0.0, 0.5, n_samples)

    # Write to DataFrame
    df = pd.DataFrame({
        "timestamp_ms": ts.astype(np.int64),
        "ir_finger":    ir_finger.astype(np.int64),
        "ir_toe":       ir_toe.astype(np.int64),
        "temp_finger":  np.round(temp_finger, 3),
        "temp_toe":     np.round(temp_toe, 3),
        "ax":           np.round(ax, 4),
        "ay":           np.round(ay, 4),
        "az":           np.round(az, 4),
        "gx":           np.round(gx, 4),
        "gy":           np.round(gy, 4),
        "gz":           np.round(gz, 4),
    })

    filename = os.path.join(raw_dir, f"{subj_id}_{label}.csv")
    df.to_csv(filename, index=False)
    print(f"Generated {filename} ({label}, HR: {heart_rate*60:.1f} bpm, "
          f"TempFinger: {temp_finger_mean:.1f}°C, TempToe: {temp_toe_mean:.1f}°C, "
          f"Duration: {duration}s)")

print(f"Synthetic raw data generation complete! Generated {num_subjects} randomized subjects.")
