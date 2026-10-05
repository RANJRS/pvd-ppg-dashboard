#pragma once
#include <cstdint>

// ---------- I2C / Sensor Wiring ----------
// Both PPG sensors share the I2C bus but sit behind a TCA9548A mux
// (finger sensor = mux ch0, toe sensor = mux ch1) since MAX3010x parts
// share a fixed I2C address.
#define I2C_SDA_PIN 8
#define I2C_SCL_PIN 9
#define MUX_ADDR 0x70
#define MUX_CH_FINGER 0
#define MUX_CH_TOE 1

// ---------- Sampling ----------
#define PPG_SAMPLE_RATE_HZ 200 // per site
#define PPG_BUFFER_SECONDS 8
#define PPG_BUFFER_LEN (PPG_SAMPLE_RATE_HZ * PPG_BUFFER_SECONDS)

// ---------- Signal Processing ----------
#define BANDPASS_LOW_HZ 0.5f
#define BANDPASS_HIGH_HZ 8.0f
#define MIN_PEAK_DISTANCE_MS 300 // ~200 bpm max physiological cap

// ---------- Feature / Inference cadence ----------
#define INFERENCE_PERIOD_MS 10000 // run risk-score inference every 10s
#define FEATURE_VEC_LEN 8

// ---------- BLE ----------
#define BLE_DEVICE_NAME "PVD-Patch"
#define BLE_SERVICE_UUID "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define BLE_FEATURE_CHAR_UUID "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
#define BLE_RISK_CHAR_UUID "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

// ---------- Risk thresholds (fallback if TFLite model unavailable) ----------
// These are heuristic backstops only; the trained model is the primary
// classifier.
#define PI_RATIO_MODERATE_THRESH 0.60f
#define PI_RATIO_HIGH_THRESH 0.35f
#define PTT_FT_MODERATE_MS 180.0f
#define PTT_FT_HIGH_MS 240.0f

enum class RiskLevel : uint8_t { NORMAL = 0, MODERATE = 1, HIGH_RISK = 2 };