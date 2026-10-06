#include "MAX30105.h"
#include <Arduino.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <Wire.h>

// =====================================================
// WIFI
// =====================================================

const char *WIFI_SSID = "Ranji";
const char *WIFI_PASSWORD = "12345678";

// Your Render API
const char *SERVER_URL = "https://pvd-ppg-dashboard.onrender.com";

// =====================================================
// I2C
// =====================================================

#define SDA_PIN 8
#define SCL_PIN 9

#define MPU6500_ADDR 0x68

// =====================================================
// DATA SETTINGS
// =====================================================

#define SAMPLE_RATE 200
#define WINDOW_SECONDS 8
#define TOTAL_SAMPLES (SAMPLE_RATE * WINDOW_SECONDS)

// =====================================================
// SENSORS
// =====================================================

MAX30105 max30102;

// =====================================================
// MPU6500 REGISTERS
// =====================================================

#define PWR_MGMT_1 0x6B
#define CONFIG_REG 0x1A
#define GYRO_CONFIG 0x1B
#define ACCEL_CONFIG 0x1C
#define ACCEL_XOUT_H 0x3B
#define WHO_AM_I_REG 0x75

// =====================================================
// CSV STORAGE
// =====================================================

String csvData;

// =====================================================
// MPU6500 FUNCTIONS
// =====================================================

void writeMPU(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

uint8_t readMPU(uint8_t reg) {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);

  Wire.requestFrom(MPU6500_ADDR, 1);

  if (Wire.available())
    return Wire.read();

  return 0;
}

bool initMPU6500() {
  uint8_t id = readMPU(WHO_AM_I_REG);

  Serial.print("MPU6500 WHO_AM_I = 0x");
  Serial.println(id, HEX);

  if (id != 0x70)
    return false;

  // Wake sensor
  writeMPU(PWR_MGMT_1, 0x00);
  delay(100);

  // Low-pass filter
  writeMPU(CONFIG_REG, 0x03);

  // Gyroscope ±500 deg/s
  writeMPU(GYRO_CONFIG, 0x08);

  // Accelerometer ±4g
  writeMPU(ACCEL_CONFIG, 0x08);

  return true;
}

bool readMPU6500(float &ax, float &ay, float &az, float &gx, float &gy,
                 float &gz) {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(ACCEL_XOUT_H);

  if (Wire.endTransmission(false) != 0)
    return false;

  Wire.requestFrom(MPU6500_ADDR, 14);

  if (Wire.available() < 14)
    return false;

  int16_t rawAx = (Wire.read() << 8) | Wire.read();
  int16_t rawAy = (Wire.read() << 8) | Wire.read();
  int16_t rawAz = (Wire.read() << 8) | Wire.read();

  // Skip temperature
  Wire.read();
  Wire.read();

  int16_t rawGx = (Wire.read() << 8) | Wire.read();
  int16_t rawGy = (Wire.read() << 8) | Wire.read();
  int16_t rawGz = (Wire.read() << 8) | Wire.read();

  // ±4g
  ax = rawAx / 8192.0;
  ay = rawAy / 8192.0;
  az = rawAz / 8192.0;

  // ±500 deg/s
  gx = rawGx / 65.5;
  gy = rawGy / 65.5;
  gz = rawGz / 65.5;

  return true;
}

// =====================================================
// WIFI
// =====================================================

bool connectWiFi() {
  Serial.println();
  Serial.println("==========================================");
  Serial.println("CONNECTING TO WIFI");
  Serial.println("==========================================");

  Serial.print("SSID: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;

  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WIFI CONNECTED");
    Serial.print("IP ADDRESS: ");
    Serial.println(WiFi.localIP());

    return true;
  }

  Serial.println("WIFI CONNECTION FAILED");

  return false;
}

// =====================================================
// SEND DATA TO DASHBOARD
// =====================================================

bool sendDataToDashboard() {
  Serial.println();
  Serial.println("==========================================");
  Serial.println("SENDING DATA TO DASHBOARD");
  Serial.println("==========================================");

  Serial.print("Payload size: ");
  Serial.print(csvData.length());
  Serial.println(" bytes");

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected.");
    return false;
  }

  WiFiClientSecure client;

  // Prototype mode.
  // Render uses HTTPS.
  client.setInsecure();

  HTTPClient https;

  Serial.println("Connecting to Render...");

  if (!https.begin(client, SERVER_URL)) {
    Serial.println("HTTPS CONNECTION FAILED");
    return false;
  }

  https.setTimeout(30000);

  // Tell Flask that we are sending raw CSV
  https.addHeader("Content-Type", "text/csv");
  https.addHeader("X-Device-ID", "PVD_PATCH_001");

  Serial.println("Uploading 1600 samples...");

  int httpCode = https.POST(csvData);

  Serial.print("HTTP response: ");
  Serial.println(httpCode);

  if (httpCode > 0) {
    String response = https.getString();

    Serial.println();
    Serial.println("SERVER RESPONSE:");
    Serial.println(response);

    https.end();

    if (httpCode >= 200 && httpCode < 300) {
      return true;
    }
  } else {
    Serial.print("HTTP error: ");
    Serial.println(https.errorToString(httpCode));
  }

  https.end();

  return false;
}

// =====================================================
// ACQUIRE 1600 SAMPLES
// =====================================================

bool acquirePPGWindow() {
  csvData = "";

  // Reserve memory to reduce fragmentation
  csvData.reserve(70000);

  // CSV header
  csvData = "SAMPLE,TIME_US,IR,RED,AX,AY,AZ,GX,GY,GZ\n";

  int sampleCount = 0;

  Serial.println();
  Serial.println("==========================================");
  Serial.println("COLLECTING 8-SECOND PPG WINDOW");
  Serial.println("==========================================");

  while (sampleCount < TOTAL_SAMPLES) {
    max30102.check();

    while (max30102.available() && sampleCount < TOTAL_SAMPLES) {
      uint32_t irValue = max30102.getFIFOIR();

      uint32_t redValue = max30102.getFIFORed();

      float ax, ay, az;
      float gx, gy, gz;

      bool motionOK = readMPU6500(ax, ay, az, gx, gy, gz);

      if (motionOK) {
        uint32_t timestamp = micros();

        // Add CSV row
        csvData += String(sampleCount);
        csvData += ",";

        csvData += String(timestamp);
        csvData += ",";

        csvData += String(irValue);
        csvData += ",";

        csvData += String(redValue);
        csvData += ",";

        csvData += String(ax, 4);
        csvData += ",";

        csvData += String(ay, 4);
        csvData += ",";

        csvData += String(az, 4);
        csvData += ",";

        csvData += String(gx, 2);
        csvData += ",";

        csvData += String(gy, 2);
        csvData += ",";

        csvData += String(gz, 2);
        csvData += "\n";

        sampleCount++;

        // Stream real-time sample for Web Serial dashboard
        Serial.printf("DATA,%lu,%lu,%lu\n", (unsigned long)millis(), (unsigned long)redValue, (unsigned long)irValue);

        // Progress
        if (sampleCount % 200 == 0) {
          Serial.printf("# Samples: %d/%d\n", sampleCount, TOTAL_SAMPLES);
        }
      }

      max30102.nextSample();
    }

    delay(1);
  }

  Serial.println();
  Serial.println("PPG WINDOW COMPLETE.");

  Serial.print("Final samples: ");
  Serial.println(sampleCount);

  Serial.print("CSV size: ");
  Serial.print(csvData.length());
  Serial.println(" bytes");

  return true;
}

// =====================================================
// SETUP
// =====================================================

void setup() {
  Serial.begin(115200);

  delay(2000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  Serial.println();
  Serial.println("==========================================");
  Serial.println("      PVD PPG SMART PATCH");
  Serial.println("==========================================");

  // -----------------------------------------
  // MAX30102
  // -----------------------------------------

  if (!max30102.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 NOT FOUND!");

    while (1)
      delay(1000);
  }

  Serial.println("MAX30102: OK");

  max30102.setup(60, 1, 2, 200, 411, 4096);

  max30102.setPulseAmplitudeRed(0x24);
  max30102.setPulseAmplitudeIR(0x24);

  // -----------------------------------------
  // MPU6500
  // -----------------------------------------

  if (!initMPU6500()) {
    Serial.println("ERROR: MPU6500 NOT FOUND!");

    while (1)
      delay(1000);
  }

  Serial.println("MPU6500: OK");

  // -----------------------------------------
  // WIFI
  // -----------------------------------------

  connectWiFi();

  Serial.println();
  Serial.println("SYSTEM READY");
}

// =====================================================
// LOOP
// =====================================================

void loop() {
  // Reconnect WiFi if necessary
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected.");
    connectWiFi();
  }

  // -----------------------------------------
  // Acquire 1600 samples
  // -----------------------------------------

  if (acquirePPGWindow()) {
    // -----------------------------------------
    // Send to Render
    // -----------------------------------------

    bool sent = sendDataToDashboard();

    if (sent) {
      Serial.println();
      Serial.println("==========================================");
      Serial.println("PPG DATA SENT SUCCESSFULLY");
      Serial.println("==========================================");
    } else {
      Serial.println();
      Serial.println("==========================================");
      Serial.println("PPG DATA SEND FAILED");
      Serial.println("==========================================");
    }
  }

  Serial.println();
  Serial.println("Starting next acquisition...");
  delay(3000);
}