#include "ble_service.h"
#include <NimBLEDevice.h>

namespace {
NimBLECharacteristic *featureChar = nullptr;
NimBLECharacteristic *riskChar = nullptr;
}  // namespace

void BleService::begin() {
    NimBLEDevice::init(BLE_DEVICE_NAME);
    NimBLEServer *server = NimBLEDevice::createServer();
    NimBLEService *service = server->createService(BLE_SERVICE_UUID);

    featureChar = service->createCharacteristic(
        BLE_FEATURE_CHAR_UUID,
        NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);

    riskChar = service->createCharacteristic(
        BLE_RISK_CHAR_UUID,
        NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);

    service->start();

    NimBLEAdvertising *advertising = NimBLEDevice::getAdvertising();
    advertising->addServiceUUID(BLE_SERVICE_UUID);
    advertising->setScanResponse(true);
    advertising->start();

    Serial.println("[BLE] Advertising as " BLE_DEVICE_NAME);
}

void BleService::notifyFeatures(const float *features, int len) {
    if (!featureChar) return;
    featureChar->setValue(reinterpret_cast<const uint8_t *>(features),
                           len * sizeof(float));
    featureChar->notify();
}

void BleService::notifyRisk(RiskLevel level) {
    if (!riskChar) return;
    uint8_t val = static_cast<uint8_t>(level);
    riskChar->setValue(&val, 1);
    riskChar->notify();
}
