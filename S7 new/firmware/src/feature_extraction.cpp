#include "feature_extraction.h"
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// PPG features (mirror of ml_pipeline/feature_extraction.py)
// ---------------------------------------------------------------------------

float computePerfusionIndex(const SiteBuffer &buf) {
    // PI = (AC amplitude / DC amplitude) * 100, averaged over detected peaks.
    if (buf.peaks.empty() || buf.dc_mean <= 0.0f) return 0.0f;

    float acSum = 0.0f;
    for (const auto &p : buf.peaks) acSum += p.amplitude;
    float acMean = acSum / buf.peaks.size();
    return (acMean / buf.dc_mean) * 100.0f;
}

float computePulseTransitTime(const SiteBuffer &proximal, const SiteBuffer &distal) {
    // PTT = time delay between corresponding systolic peaks at the finger
    // (proximal) vs toe (distal) sites.
    if (proximal.peaks.empty() || distal.peaks.empty()) return 0.0f;

    std::vector<float> deltas;
    for (const auto &dp : distal.peaks) {
        uint32_t bestDelta = UINT32_MAX;
        for (const auto &pp : proximal.peaks) {
            if (dp.timestamp_ms <= pp.timestamp_ms) continue;
            uint32_t delta = dp.timestamp_ms - pp.timestamp_ms;
            if (delta < 50 || delta > 400) continue;
            if (delta < bestDelta) bestDelta = delta;
        }
        if (bestDelta != UINT32_MAX) deltas.push_back(static_cast<float>(bestDelta));
    }

    if (deltas.empty()) return 0.0f;
    float sum = std::accumulate(deltas.begin(), deltas.end(), 0.0f);
    return sum / deltas.size();
}

float computeAugmentationIndex(const SiteBuffer &buf) {
    // AI = (dicrotic notch amplitude / systolic peak amplitude) * 100.
    if (buf.peaks.empty()) return 0.0f;

    float sum = 0.0f;
    int count = 0;
    for (const auto &p : buf.peaks) {
        if (p.amplitude <= 0.0f) continue;
        sum += (p.dicrotic_notch_amplitude / p.amplitude) * 100.0f;
        count++;
    }
    return count > 0 ? sum / count : 0.0f;
}

float computeHrvRmssd(const SiteBuffer &buf) {
    // RMSSD of successive inter-beat intervals (from finger peaks).
    if (buf.peaks.size() < 3) return 0.0f;

    std::vector<float> ibi;
    for (size_t i = 1; i < buf.peaks.size(); ++i) {
        ibi.push_back(static_cast<float>(
            buf.peaks[i].timestamp_ms - buf.peaks[i - 1].timestamp_ms));
    }

    float sumSqDiff = 0.0f;
    for (size_t i = 1; i < ibi.size(); ++i) {
        float diff = ibi[i] - ibi[i - 1];
        sumSqDiff += diff * diff;
    }
    return std::sqrt(sumSqDiff / (ibi.size() - 1));
}

float computeDicroticRatio(const SiteBuffer &buf) {
    if (buf.peaks.empty()) return 0.0f;
    float sum = 0.0f;
    int count = 0;
    for (const auto &p : buf.peaks) {
        if (p.amplitude <= 0.0f) continue;
        sum += p.dicrotic_notch_amplitude / p.amplitude;
        count++;
    }
    return count > 0 ? sum / count : 0.0f;
}

// ---------------------------------------------------------------------------
// Temperature features
// ---------------------------------------------------------------------------

static float vectorMean(const std::vector<float> &v) {
    if (v.empty()) return 0.0f;
    float s = 0.0f;
    for (float x : v) s += x;
    return s / v.size();
}

// ---------------------------------------------------------------------------
// IMU features
// ---------------------------------------------------------------------------

float computeAccelStd(const ImuBuffer &imu) {
    // Motion artefact index: std-dev of 3-axis acceleration magnitude.
    size_t n = imu.ax.size();
    if (n == 0) return 0.0f;

    std::vector<float> mag(n);
    for (size_t i = 0; i < n; ++i) {
        mag[i] = std::sqrt(imu.ax[i]*imu.ax[i] +
                           imu.ay[i]*imu.ay[i] +
                           imu.az[i]*imu.az[i]);
    }
    // Compute mean then std
    float mean = vectorMean(mag);
    float varSum = 0.0f;
    for (float m : mag) varSum += (m - mean) * (m - mean);
    return std::sqrt(varSum / n);
}

float computeGyroStd(const ImuBuffer &imu) {
    // Rotational motion index: std-dev of 3-axis gyroscope magnitude.
    size_t n = imu.gx.size();
    if (n == 0) return 0.0f;

    std::vector<float> mag(n);
    for (size_t i = 0; i < n; ++i) {
        mag[i] = std::sqrt(imu.gx[i]*imu.gx[i] +
                           imu.gy[i]*imu.gy[i] +
                           imu.gz[i]*imu.gz[i]);
    }
    float mean = vectorMean(mag);
    float varSum = 0.0f;
    for (float m : mag) varSum += (m - mean) * (m - mean);
    return std::sqrt(varSum / n);
}

// ---------------------------------------------------------------------------
// Main feature extractor — produces FEATURE_VEC_LEN = 13 features
// ---------------------------------------------------------------------------

void extractFeatures(const SiteBuffer  &fingerBuf,
                     const SiteBuffer  &toeBuf,
                     const TempBuffer  &tempBuf,
                     const ImuBuffer   &imuBuf,
                     float             *out) {
    // --- PPG features (0-7) ---
    float pi_finger = computePerfusionIndex(fingerBuf);
    float pi_toe    = computePerfusionIndex(toeBuf);
    float pi_ratio  = (pi_finger > 1e-3f) ? (pi_toe / pi_finger) : 0.0f;

    out[0] = pi_finger;
    out[1] = pi_toe;
    out[2] = pi_ratio;
    out[3] = computePulseTransitTime(fingerBuf, toeBuf);
    out[4] = computeAugmentationIndex(fingerBuf);
    out[5] = computeAugmentationIndex(toeBuf);
    out[6] = computeHrvRmssd(fingerBuf);
    out[7] = computeDicroticRatio(toeBuf);

    // --- Temperature features (8-10) ---
    float t_finger  = vectorMean(tempBuf.finger_raw);
    float t_toe     = vectorMean(tempBuf.toe_raw);
    out[8]  = t_finger;
    out[9]  = t_toe;
    out[10] = t_finger - t_toe;   // temp gradient (>0 = finger warmer than toe)

    // --- IMU features (11-12) ---
    out[11] = computeAccelStd(imuBuf);
    out[12] = computeGyroStd(imuBuf);
}
