#pragma once
#include <Arduino.h>
#include "MAX30105.h"
#include "config.h"

// Wraps two MAX3010x sensors (finger + toe) behind a TCA9548A I2C mux.
// Provides synchronized-ish sampling by round-robining the mux at
// PPG_SAMPLE_RATE_HZ; drift between sites is corrected downstream in
// feature_extraction using timestamp interpolation.

struct PpgSample {
    uint32_t ir_finger;
    uint32_t ir_toe;
    uint32_t timestamp_ms;
};

class PpgSensor {
public:
    bool begin();
    // Blocking-ish read; call from the sampling task at PPG_SAMPLE_RATE_HZ.
    PpgSample readSample();

private:
    MAX30105 finger_;
    MAX30105 toe_;
    void selectMuxChannel(uint8_t ch);
    bool initSite(MAX30105 &sensor, uint8_t ch);
};
