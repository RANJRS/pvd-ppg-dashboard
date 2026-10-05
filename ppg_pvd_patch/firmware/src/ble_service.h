#pragma once
#include "config.h"

// Thin NimBLE-based GATT service:
//  - FEATURE characteristic: notifies the raw 8-float feature vector
//  - RISK characteristic: notifies a single byte risk level (0/1/2)
// A companion phone app / gateway subscribes to both; RISK alone is enough
// to drive a buzzer/LED alert locally on the patch too.
class BleService {
public:
    void begin();
    void notifyFeatures(const float *features, int len);
    void notifyRisk(RiskLevel level);
};
