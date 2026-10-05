#pragma once
#include <vector>
#include "signal_processing.h"
#include "config.h"

struct SiteBuffer {
    std::vector<float>    raw;
    std::vector<float>    filtered;
    std::vector<uint32_t> timestamps_ms;
    std::vector<PeakInfo> peaks;
    float dc_mean = 0.0f;   // DC component, needed for perfusion index
};

// IMU circular buffer (same rolling-window as PPG buffers)
struct ImuBuffer {
    std::vector<float> ax, ay, az;   // acceleration m/s²
    std::vector<float> gx, gy, gz;   // gyroscope °/s
};

// Temperature: a single scalar per site, updated each sample
struct TempBuffer {
    float finger_mean = 0.0f;   // running mean, °C
    float toe_mean    = 0.0f;
    // Raw history needed to compute the mean in inferenceTask
    std::vector<float> finger_raw;
    std::vector<float> toe_raw;
};

// Computes the FEATURE_VEC_LEN (= 13) feature vector shared with the Python
// training pipeline (see ml_pipeline/feature_extraction.py — keep in sync).
// out must point to a buffer of size FEATURE_VEC_LEN.
//
// Feature layout:
//   [0]  PI_finger       [1]  PI_toe         [2]  PI_ratio
//   [3]  PTT_ft (ms)     [4]  AI_finger      [5]  AI_toe
//   [6]  HRV_rmssd       [7]  dicrotic_ratio (toe)
//   [8]  temp_finger     [9]  temp_toe       [10] temp_diff
//   [11] accel_std       [12] gyro_std
void extractFeatures(const SiteBuffer  &fingerBuf,
                     const SiteBuffer  &toeBuf,
                     const TempBuffer  &tempBuf,
                     const ImuBuffer   &imuBuf,
                     float             *out);

// Individual feature helpers, exposed for unit testing.
float computePerfusionIndex(const SiteBuffer &buf);
float computePulseTransitTime(const SiteBuffer &proximal, const SiteBuffer &distal);
float computeAugmentationIndex(const SiteBuffer &buf);
float computeHrvRmssd(const SiteBuffer &buf);
float computeDicroticRatio(const SiteBuffer &buf);
float computeAccelStd(const ImuBuffer &imu);
float computeGyroStd(const ImuBuffer &imu);
