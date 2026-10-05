#pragma once
#include <vector>
#include "signal_processing.h"
#include "config.h"

struct SiteBuffer {
    std::vector<float> raw;
    std::vector<float> filtered;
    std::vector<uint32_t> timestamps_ms;
    std::vector<PeakInfo> peaks;
    float dc_mean = 0.0f;   // DC component, needed for perfusion index
};

// Computes the FEATURE_VEC_LEN feature vector shared with the Python
// training pipeline (see ml_pipeline/feature_extraction.py — keep in sync).
// out must point to a buffer of size FEATURE_VEC_LEN.
void extractFeatures(const SiteBuffer &fingerBuf, const SiteBuffer &toeBuf,
                      float *out);

// Individual feature helpers, exposed for unit testing.
float computePerfusionIndex(const SiteBuffer &buf);
float computePulseTransitTime(const SiteBuffer &proximal, const SiteBuffer &distal);
float computeAugmentationIndex(const SiteBuffer &buf);
float computeHrvRmssd(const SiteBuffer &buf);
float computeDicroticRatio(const SiteBuffer &buf);
