#pragma once
#include <Arduino.h>
#include "MAX30105.h"
#include "config.h"

// Wraps two MAX3010x sensors (finger + toe) behind a TCA9548A I2C mux.
// Provides synchronized-ish sampling by round-robining the mux at
// PPG_SAMPLE_RATE_HZ; drift between sites is corrected downstream in
// feature_extraction using timestamp interpolation.
//
// Temperature readings are taken at each site using the built-in
// MAX30105 die-temperature register (NTC thermistor, ~0.0625°C resolution).
// IMU data (6-DOF: ax,ay,az,gx,gy,gz) is read from a secondary sensor
// (e.g. MPU-6050 on the same I2C bus); see ppg_sensor.cpp.

struct PpgSample {
    uint32_t ir_finger;
    uint32_t ir_toe;
    uint32_t timestamp_ms;
    // Skin temperature (°C), one reading per site per sample
    float    temp_finger;     // from MAX30105 NTC or external thermistor
    float    temp_toe;
    // 6-DOF IMU — acceleration (m/s²) + gyroscope (°/s)
    float    ax, ay, az;     // accelerometer axes
    float    gx, gy, gz;     // gyroscope axes
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
