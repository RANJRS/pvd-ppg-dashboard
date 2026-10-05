#include "signal_processing.h"
#include <cmath>
#include <algorithm>

BiquadFilter makeBandpassFilter() {
    // Coefficients for fs=200Hz, Butterworth bandpass 0.5-8.0Hz, order 2.
    // Generated via scipy.signal.butter(2, [0.5, 8.0], btype='bandpass', fs=200)
    // then converted to a single biquad (SOS[0]) — regenerate with
    // ml_pipeline/data_preprocessing.py if sample rate or band changes.
    BiquadFilter f;
    f.b0 = 0.1367f;
    f.b1 = 0.0f;
    f.b2 = -0.1367f;
    f.a1 = -1.6835f;
    f.a2 = 0.7265f;
    return f;
}

std::vector<PeakInfo> detectPeaks(const std::vector<float> &filtered,
                                   const std::vector<uint32_t> &timestamps_ms) {
    std::vector<PeakInfo> peaks;
    if (filtered.size() < 3) return peaks;

    const size_t minDistanceSamples =
        (MIN_PEAK_DISTANCE_MS * PPG_SAMPLE_RATE_HZ) / 1000;

    // Adaptive threshold: mean + 0.5*std over the buffer, recalculated
    // per-call since perfusion (and hence amplitude) drifts over the session.
    float mean = 0.0f;
    for (float v : filtered) mean += v;
    mean /= filtered.size();

    float variance = 0.0f;
    for (float v : filtered) variance += (v - mean) * (v - mean);
    variance /= filtered.size();
    float threshold = mean + 0.5f * std::sqrt(variance);

    size_t lastPeakIdx = 0;
    bool hasLastPeak = false;

    for (size_t i = 1; i + 1 < filtered.size(); ++i) {
        bool isLocalMax = filtered[i] > filtered[i - 1] && filtered[i] >= filtered[i + 1];
        if (isLocalMax && filtered[i] > threshold) {
            if (hasLastPeak && (i - lastPeakIdx) < minDistanceSamples) continue;

            PeakInfo p;
            p.index = i;
            p.timestamp_ms = timestamps_ms[i];
            p.amplitude = filtered[i];

            // Dicrotic notch: local minimum followed by a smaller secondary
            // peak within ~150-400ms after the systolic peak. Search that
            // window; if found, record its amplitude for the AI/dicrotic
            // ratio features. If absent (common in stiff/low-perfusion
            // vessels), amplitude is reported as 0.
            size_t searchEnd = std::min(filtered.size(),
                i + (400 * PPG_SAMPLE_RATE_HZ) / 1000);
            size_t searchStart = i + (150 * PPG_SAMPLE_RATE_HZ) / 1000;
            float dicroticAmp = 0.0f;
            for (size_t j = searchStart; j + 1 < searchEnd && j > 0; ++j) {
                bool secondaryMax = filtered[j] > filtered[j - 1] &&
                                     filtered[j] >= filtered[j + 1];
                if (secondaryMax) {
                    dicroticAmp = filtered[j];
                    break;
                }
            }
            p.dicrotic_notch_amplitude = dicroticAmp;

            peaks.push_back(p);
            lastPeakIdx = i;
            hasLastPeak = true;
        }
    }
    return peaks;
}
