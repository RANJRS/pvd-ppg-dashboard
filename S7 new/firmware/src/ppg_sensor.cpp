#include "ppg_sensor.h"
#include <Wire.h>

// ---------- Helper: select I2C mux channel ----------
void PpgSensor::selectMuxChannel(uint8_t ch) {
    Wire.beginTransmission(MUX_ADDR);
    Wire.write(1 << ch);
    Wire.endTransmission();
}

// ---------- Helper: initialise one PPG site ----------
bool PpgSensor::initSite(MAX30105 &sensor, uint8_t ch) {
    selectMuxChannel(ch);
    if (!sensor.begin(Wire, I2C_SPEED_FAST)) return false;

    // Config tuned for low-perfusion peripheral sites (toe especially):
    // higher LED current + averaging to keep SNR usable on diabetic patients
    // who often have reduced peripheral blood flow.
    byte ledBrightness = 0x3F;   // ~12.6 mA
    byte sampleAverage = 4;
    byte ledMode       = 2;      // Red + IR
    int  sampleRate    = PPG_SAMPLE_RATE_HZ;
    int  pulseWidth    = 411;
    int  adcRange      = 4096;

    sensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
    return true;
}

// ---------- Public: initialise all sensors ----------
bool PpgSensor::begin() {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(400000);

    bool okFinger = initSite(finger_, MUX_CH_FINGER);
    bool okToe    = initSite(toe_,    MUX_CH_TOE);

    if (!okFinger) Serial.println("[PPG] Finger sensor init FAILED");
    if (!okToe)    Serial.println("[PPG] Toe sensor init FAILED");

    // NOTE: If an external IMU (e.g. MPU-6050 at 0x68) is wired to the same
    // I2C bus, call its begin() here and set a flag.  The stubs below will
    // return zeros until you wire up the real driver.

    return okFinger && okToe;
}

// ---------- Public: read one sample from all sensors ----------
PpgSample PpgSensor::readSample() {
    PpgSample s;
    s.timestamp_ms = millis();

    // --- PPG ---
    selectMuxChannel(MUX_CH_FINGER);
    s.ir_finger = finger_.getIR();

    selectMuxChannel(MUX_CH_TOE);
    s.ir_toe = toe_.getIR();

    // --- Skin Temperature ---
    // MAX30105 has a built-in die-temperature sensor (NTC-based).
    // It reads the die temperature, which correlates well with contact skin
    // temperature for a wrist/finger patch application.
    //
    // Call readTemperature() after each site read while that mux channel is
    // still selected.  The function triggers an ADC conversion and returns °C.
    selectMuxChannel(MUX_CH_FINGER);
    s.temp_finger = finger_.readTemperature();

    selectMuxChannel(MUX_CH_TOE);
    s.temp_toe = toe_.readTemperature();

    // --- IMU (6-DOF) ---
    // Stub: returns zeros until you connect an IMU driver.
    // Replace the lines below with your MPU-6050/ICM-42688 driver calls.
    // Example (MPU-6050 via Wire):
    //   imu_.getAcceleration(&s.ax, &s.ay, &s.az);
    //   imu_.getRotation(&s.gx, &s.gy, &s.gz);
    s.ax = 0.0f;  s.ay = 0.0f;  s.az = 9.81f;
    s.gx = 0.0f;  s.gy = 0.0f;  s.gz = 0.0f;

    return s;
}
