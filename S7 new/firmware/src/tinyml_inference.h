#pragma once
#include "config.h"

// Wraps a TFLite Micro interpreter running the 8-feature -> 3-class
// (Normal/Moderate/High) MLP exported by ml_pipeline/convert_to_tflite.py.
class TinyMlInference {
public:
    bool begin();
    // Runs inference on an 8-element feature vector, returns risk level
    // and writes class probabilities into probsOut[3] if non-null.
    RiskLevel predict(const float *features, float *probsOut = nullptr);

    // Heuristic fallback used only if the model failed to load (e.g. flash
    // corruption) — keeps the device safe-failing rather than silent.
    RiskLevel predictHeuristic(const float *features);

private:
    bool modelLoaded_ = false;
    // Opaque TFLite Micro state (interpreter, arena, resolver) lives in the
    // .cpp to keep this header framework-agnostic for unit tests.
};
