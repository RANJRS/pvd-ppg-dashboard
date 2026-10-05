#pragma once
#include <cstdint>
#include <vector>
#include "config.h"

// 2nd-order Butterworth bandpass, coefficients pre-computed offline for
// fs=200Hz, [0.5, 8.0] Hz (see ml_pipeline/data_preprocessing.py for the
// scipy design that generated these — kept identical so firmware and
// training-time filtering match).
struct BiquadFilter {
    float b0, b1, b2, a1, a2;
    float z1 = 0.0f, z2 = 0.0f;

    float process(float x) {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
};

BiquadFilter makeBandpassFilter();

struct PeakInfo {
    size_t index;
    uint32_t timestamp_ms;
    float amplitude;
    float dicrotic_notch_amplitude;  // amplitude at secondary (dicrotic) peak
};

// Detects systolic peaks + dicrotic notches in a filtered PPG buffer.
std::vector<PeakInfo> detectPeaks(const std::vector<float> &filtered,
                                   const std::vector<uint32_t> &timestamps_ms);
