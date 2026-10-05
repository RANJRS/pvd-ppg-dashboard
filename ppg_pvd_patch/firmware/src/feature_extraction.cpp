#include "feature_extraction.h"
#include <cmath>
#include <numeric>

float computePerfusionIndex(const SiteBuffer &buf) {
    // PI = (AC amplitude / DC amplitude) * 100, averaged over detected peaks.
    // Standard clinical PPG perfusion index definition.
    if (buf.peaks.empty() || buf.dc_mean <= 0.0f) return 0.0f;

    float acSum = 0.0f;
    for (const auto &p : buf.peaks) acSum += p.amplitude;
    float acMean = acSum / buf.peaks.size();

    return (acMean / buf.dc_mean) * 100.0f;
}

float computePulseTransitTime(const SiteBuffer &proximal, const SiteBuffer &distal) {
    // PTT = time delay between corresponding systolic peaks at the finger
    // (proximal) vs toe (distal) sites. Match each distal peak to the
    // nearest-preceding proximal peak within a physiologically plausible
    // window (50-400ms), then average.
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
    // Elevated AI indicates arterial stiffening — earlier reflected wave
    // return merges into the systolic peak rather than appearing as a
    // distinct dicrotic notch.
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
    // RMSSD of successive inter-beat intervals (from finger peaks — the
    // higher-SNR site). Reduced HRV is an independent diabetic autonomic
    // neuropathy marker, complements the vascular features.
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

void extractFeatures(const SiteBuffer &fingerBuf, const SiteBuffer &toeBuf,
                      float *out) {
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
}
