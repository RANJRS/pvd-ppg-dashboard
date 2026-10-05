#include "ppg_sensor.h"
#include <Wire.h>

void PpgSensor::selectMuxChannel(uint8_t ch) {
    Wire.beginTransmission(MUX_ADDR);
    Wire.write(1 << ch);
    Wire.endTransmission();
}

bool PpgSensor::initSite(MAX30105 &sensor, uint8_t ch) {
    selectMuxChannel(ch);
    if (!sensor.begin(Wire, I2C_SPEED_FAST)) return false;

    // Config tuned for low-perfusion peripheral sites (toe especially):
    // higher LED current + averaging to keep SNR usable on diabetic patients
    // who often have reduced peripheral blood flow.
    byte ledBrightness = 0x3F;   // ~12.6mA
    byte sampleAverage  = 4;
    byte ledMode        = 2;     // Red + IR
    int  sampleRate     = PPG_SAMPLE_RATE_HZ;
    int  pulseWidth     = 411;
    int  adcRange       = 4096;

    sensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
    return true;
}

bool PpgSensor::begin() {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(400000);

    bool okFinger = initSite(finger_, MUX_CH_FINGER);
    bool okToe    = initSite(toe_, MUX_CH_TOE);

    if (!okFinger) Serial.println("[PPG] Finger sensor init FAILED");
    if (!okToe)    Serial.println("[PPG] Toe sensor init FAILED");

    return okFinger && okToe;
}

PpgSample PpgSensor::readSample() {
    PpgSample s;
    s.timestamp_ms = millis();

    selectMuxChannel(MUX_CH_FINGER);
    s.ir_finger = finger_.getIR();

    selectMuxChannel(MUX_CH_TOE);
    s.ir_toe = toe_.getIR();

    return s;
}
