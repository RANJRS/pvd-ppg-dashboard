#include "MAX30105.h"

#include <Arduino.h>
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <Wire.h>

// =====================================================
// BLUETOOTH BLE (PVD-Patch Nordic UART Service)
// =====================================================

#define BLE_DEVICE_NAME "PVD-Patch"
#define BLE_SERVICE_UUID "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define BLE_RX_CHAR_UUID "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
#define BLE_TX_CHAR_UUID "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

BLEServer *pBleServer = nullptr;
BLECharacteristic *pTxChar = nullptr;
BLECharacteristic *pRxChar = nullptr;
bool bleConnected = false;
volatile bool bleStartRequested = false;
volatile bool bleStopRequested = false;

class MyBleServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) {
    bleConnected = true;
    Serial.println("\n[BLE] Web Dashboard paired & connected over Bluetooth!");
  }
  void onDisconnect(BLEServer *pServer) {
    bleConnected = false;
    Serial.println(
        "\n[BLE] Dashboard disconnected. Resuming BLE advertising...");
    BLEDevice::startAdvertising();
  }
};

class MyBleRxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pChar) {
    String rxVal = pChar->getValue().c_str();
    rxVal.trim();
    rxVal.toUpperCase();
    Serial.print("[BLE RX]: ");
    Serial.println(rxVal);
    if (rxVal.startsWith("START") || rxVal == "S" || rxVal == "REC") {
      bleStartRequested = true;
    } else if (rxVal.startsWith("STOP") || rxVal == "Q") {
      bleStopRequested = true;
    }
  }
};

void initBLE() {
  BLEDevice::init(BLE_DEVICE_NAME);
  pBleServer = BLEDevice::createServer();
  pBleServer->setCallbacks(new MyBleServerCallbacks());

  BLEService *pService = pBleServer->createService(BLE_SERVICE_UUID);

  pTxChar = pService->createCharacteristic(BLE_TX_CHAR_UUID,
                                           BLECharacteristic::PROPERTY_NOTIFY);
  pTxChar->addDescriptor(new BLE2902());

  pRxChar = pService->createCharacteristic(
      BLE_RX_CHAR_UUID,
      BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  pRxChar->setCallbacks(new MyBleRxCallbacks());

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(BLE_SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("[BLE] Advertising active as 'PVD-Patch' (Ready to pair)!");
}

// =====================================================
// WIFI
// =====================================================

// KEEP YOUR EXISTING WIFI DETAILS HERE
const char *WIFI_SSID = "Ranji";
const char *WIFI_PASSWORD = "12345678";

const char *SERVER_URL = "https://pvd-ppg-dashboard.onrender.com/api/ppg";

// =====================================================
// I2C BUS 0
// Finger MAX30102 + MPU6500
// =====================================================

#define SDA_PIN 8
#define SCL_PIN 9

#define MPU6500_ADDR ((uint8_t)0x68)

// =====================================================
// I2C BUS 1
// Toe MAX30100
// =====================================================

#define TOE_SDA 4
#define TOE_SCL 5
#define MAX30100_ADDR ((uint8_t)0x57)

TwoWire WireToe = TwoWire(1);

// =====================================================
// DATA SETTINGS
// =====================================================

#define SAMPLE_RATE 200
#define WINDOW_SECONDS 15
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
// CSV
// =====================================================

String csvData;

// =====================================================
// MAX30100 REGISTERS
// =====================================================

#define MAX30100_FIFO_WR_PTR 0x02
#define MAX30100_FIFO_OVF_CTR 0x03
#define MAX30100_FIFO_RD_PTR 0x04
#define MAX30100_FIFO_DATA 0x05
#define MAX30100_MODE_CONFIG 0x06
#define MAX30100_SPO2_CONFIG 0x07
#define MAX30100_LED_CONFIG 0x09
#define MAX30100_PART_ID 0xFF

// =====================================================
// MAX30100 LOW LEVEL FUNCTIONS
// =====================================================

void max30100Write(uint8_t reg, uint8_t value) {
  WireToe.beginTransmission(MAX30100_ADDR);
  WireToe.write(reg);
  WireToe.write(value);
  WireToe.endTransmission();
}

uint8_t max30100Read(uint8_t reg) {
  WireToe.beginTransmission(MAX30100_ADDR);
  WireToe.write(reg);

  if (WireToe.endTransmission(false) != 0)
    return 0;

  WireToe.requestFrom(MAX30100_ADDR, (uint8_t)1);

  if (WireToe.available())
    return WireToe.read();

  return 0;
}

bool initMAX30100() {
  Serial.println();
  Serial.println("Initializing MAX30100...");

  WireToe.begin(TOE_SDA, TOE_SCL, 400000);

  delay(50);

  uint8_t partID = max30100Read(MAX30100_PART_ID);

  Serial.print("MAX30100 PART ID = 0x");
  Serial.println(partID, HEX);

  if (partID != 0x11) {
    Serial.println("MAX30100 NOT DETECTED");
    return false;
  }

  // Reset
  max30100Write(MAX30100_MODE_CONFIG, 0x40);
  delay(20);

  // SPO2 + Heart Rate mode
  max30100Write(MAX30100_MODE_CONFIG, 0x03);

  // 200 Hz + 800 us pulse width + high resolution
  //
  // bits 4:2 = 011 -> 200 Hz
  // bits 1:0 = 10  -> 800 us
  // bit 6     = 1   -> high resolution
  //
  max30100Write(MAX30100_SPO2_CONFIG, 0x6E);

  // IR = 27.1 mA
  // RED = 27.1 mA
  max30100Write(MAX30100_LED_CONFIG, 0x88);

  // Clear FIFO
  max30100Write(MAX30100_FIFO_WR_PTR, 0x00);
  max30100Write(MAX30100_FIFO_OVF_CTR, 0x00);
  max30100Write(MAX30100_FIFO_RD_PTR, 0x00);

  Serial.println("MAX30100 INITIALIZED");
  Serial.println("MAX30100: 200 Hz");
  Serial.println("MAX30100: 800 us pulse width");

  return true;
}

// =====================================================
// READ LATEST MAX30100 SAMPLE
// =====================================================

bool readMAX30100(uint16_t &ir, uint16_t &red) {
  uint8_t writePtr = max30100Read(MAX30100_FIFO_WR_PTR);

  uint8_t readPtr = max30100Read(MAX30100_FIFO_RD_PTR);

  uint8_t count = (writePtr - readPtr) & 0x0F;

  if (count == 0)
    return false;

  uint8_t buffer[64];

  uint8_t bytesToRead = count * 4;

  WireToe.beginTransmission(MAX30100_ADDR);
  WireToe.write(MAX30100_FIFO_DATA);

  if (WireToe.endTransmission(false) != 0)
    return false;

  WireToe.requestFrom(MAX30100_ADDR, bytesToRead);

  uint8_t received = 0;

  while (WireToe.available() && received < bytesToRead) {
    buffer[received++] = WireToe.read();
  }

  if (received < 4)
    return false;

  // Use the newest sample
  uint8_t index = (count - 1) * 4;

  ir = ((uint16_t)buffer[index] << 8) | buffer[index + 1];

  red = ((uint16_t)buffer[index + 2] << 8) | buffer[index + 3];

  return true;
}

// =====================================================
// MPU6500
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

  Wire.requestFrom(MPU6500_ADDR, (uint8_t)1);

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

  writeMPU(PWR_MGMT_1, 0x00);
  delay(100);

  writeMPU(CONFIG_REG, 0x03);

  // ±500 dps
  writeMPU(GYRO_CONFIG, 0x08);

  // ±4g
  writeMPU(ACCEL_CONFIG, 0x08);

  return true;
}

bool readMPU6500(float &ax, float &ay, float &az, float &gx, float &gy,
                 float &gz) {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(ACCEL_XOUT_H);

  if (Wire.endTransmission(false) != 0)
    return false;

  Wire.requestFrom(MPU6500_ADDR, (uint8_t)14);

  if (Wire.available() < 14)
    return false;

  int16_t rawAx = (Wire.read() << 8) | Wire.read();

  int16_t rawAy = (Wire.read() << 8) | Wire.read();

  int16_t rawAz = (Wire.read() << 8) | Wire.read();

  // Temperature
  Wire.read();
  Wire.read();

  int16_t rawGx = (Wire.read() << 8) | Wire.read();

  int16_t rawGy = (Wire.read() << 8) | Wire.read();

  int16_t rawGz = (Wire.read() << 8) | Wire.read();

  // ±4g
  ax = rawAx / 8192.0;
  ay = rawAy / 8192.0;
  az = rawAz / 8192.0;

  // ±500 dps
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

  WiFi.mode(WIFI_STA);

  WiFi.config(INADDR_NONE, INADDR_NONE, INADDR_NONE, IPAddress(8, 8, 8, 8),
              IPAddress(1, 1, 1, 1));

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;

  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WIFI CONNECTION FAILED");
    return false;
  }

  Serial.println("WIFI CONNECTED");

  Serial.print("IP ADDRESS: ");
  Serial.println(WiFi.localIP());

  return true;
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

  client.setInsecure();

  Serial.println("Connecting to Render...");

  if (!client.connect("pvd-ppg-dashboard.onrender.com", 443)) {
    Serial.println("HTTPS CONNECTION FAILED");
    client.stop();
    return false;
  }

  Serial.println("Render connected.");

  client.print("POST /api/ppg HTTP/1.1\r\n");

  client.print("Host: pvd-ppg-dashboard.onrender.com\r\n");

  client.print("Content-Type: text/csv\r\n");

  client.print("X-Device-ID: PVD_PATCH_001\r\n");

  client.print("Content-Length: ");
  client.print(csvData.length());
  client.print("\r\n");

  client.print("Connection: close\r\n\r\n");

  const size_t CHUNK_SIZE = 1024;

  size_t total = csvData.length();
  size_t sent = 0;

  while (sent < total) {
    size_t remaining = total - sent;

    size_t chunk = remaining > CHUNK_SIZE ? CHUNK_SIZE : remaining;

    size_t written =
        client.write((const uint8_t *)csvData.c_str() + sent, chunk);

    if (written != chunk) {
      Serial.println("ERROR: CSV upload failed");

      client.stop();
      return false;
    }

    sent += written;

    if (sent % 10240 < CHUNK_SIZE || sent == total) {
      Serial.print("Uploaded: ");
      Serial.print(sent);
      Serial.print("/");
      Serial.println(total);
    }

    delay(2);
  }

  Serial.println("CSV upload complete.");

  unsigned long timeout = millis();

  while (!client.available()) {
    if (millis() - timeout > 30000) {
      Serial.println("SERVER RESPONSE TIMEOUT");

      client.stop();
      return false;
    }

    delay(10);
  }

  String statusLine = client.readStringUntil('\n');

  statusLine.trim();

  Serial.print("HTTP STATUS: ");
  Serial.println(statusLine);

  while (client.connected()) {
    String line = client.readStringUntil('\n');

    if (line == "\r" || line.length() == 0) {
      break;
    }
  }

  String response = "";

  while (client.available()) {
    response += client.readString();
  }

  Serial.println();
  Serial.println("SERVER RESPONSE:");
  Serial.println(response);

  client.stop();

  if (statusLine.indexOf("200") >= 0 || statusLine.indexOf("201") >= 0 ||
      statusLine.indexOf("202") >= 0) {
    Serial.println();
    Serial.println("PPG DATA SENT SUCCESSFULLY");

    return true;
  }

  Serial.println("SERVER REJECTED DATA");

  return false;
}

// =====================================================
// SENSOR POWER MANAGEMENT (STANDBY / ACTIVE)
// =====================================================

void turnOffMAX30102() {
  max30102.setPulseAmplitudeRed(0);
  max30102.setPulseAmplitudeIR(0);
  max30102.setPulseAmplitudeGreen(0);
  max30102.shutDown();
}

void turnOnMAX30102() {
  max30102.wakeUp();
  // 200 Hz with sampleAverage = 1 (true 200 Hz output)
  max30102.setup(60, 1, 2, 200, 411, 4096);
  max30102.setPulseAmplitudeGreen(0);
  max30102.clearFIFO();
}

void turnOffMAX30100() {
  max30100Write(MAX30100_LED_CONFIG, 0x00);
  max30100Write(MAX30100_MODE_CONFIG, 0x80); // SHDN = 1
}

void turnOnMAX30100() {
  max30100Write(MAX30100_MODE_CONFIG, 0x03);
  max30100Write(MAX30100_SPO2_CONFIG, 0x6E);
  max30100Write(MAX30100_LED_CONFIG, 0x88);
  max30100Write(MAX30100_FIFO_WR_PTR, 0x00);
  max30100Write(MAX30100_FIFO_OVF_CTR, 0x00);
  max30100Write(MAX30100_FIFO_RD_PTR, 0x00);
}

void setSensorsPower(bool enable) {
  if (enable) {
    turnOnMAX30102();
    turnOnMAX30100();
    delay(50);
  } else {
    turnOffMAX30102();
    turnOffMAX30100();
  }
}

// =====================================================
// COMMAND & TRIGGER CHECKS
// =====================================================

bool checkSerialStart() {
  while (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();
    if (cmd.startsWith("START") || cmd == "S" || cmd == "REC") {
      return true;
    }
  }
  return false;
}

bool checkSerialStop() {
  while (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();
    if (cmd.startsWith("STOP") || cmd == "Q") {
      return true;
    }
  }
  return false;
}

bool checkServerStart() {
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  WiFiClientSecure client;
  client.setInsecure();
  client.setTimeout(2000);

  if (!client.connect("pvd-ppg-dashboard.onrender.com", 443)) {
    return false;
  }

  client.print("GET /api/recording/status HTTP/1.1\r\n"
               "Host: pvd-ppg-dashboard.onrender.com\r\n"
               "Connection: close\r\n\r\n");

  unsigned long start = millis();
  while (!client.available() && millis() - start < 2000) {
    delay(10);
  }

  String response = "";
  while (client.available()) {
    response += client.readString();
  }
  client.stop();

  if (response.indexOf("\"recording\":true") >= 0 ||
      response.indexOf("\"recording\": true") >= 0) {
    return true;
  }

  return false;
}

// =====================================================
// ACQUIRE 1600 SAMPLES (ON DEMAND ONLY)
// =====================================================

bool acquirePPGWindow() {
  // Power sensors ON
  setSensorsPower(true);

  csvData = "";

  // Reserve enough memory for 15s of CSV data (3000 lines)
  csvData.reserve(250000);

  csvData = "SAMPLE,TIME_US,"
            "FINGER_IR,FINGER_RED,"
            "TOE_IR,TOE_RED,"
            "AX,AY,AZ,GX,GY,GZ\n";

  int sampleCount = 0;

  uint16_t lastToeIR = 0;
  uint16_t lastToeRED = 0;

  // Signal starting status to Web Serial and BLE
  Serial.println("STATUS,RECORDING");
  if (bleConnected && pTxChar) {
    pTxChar->setValue((uint8_t *)"STATUS,RECORDING\n", 17);
    pTxChar->notify();
  }

  Serial.println();
  Serial.println("==========================================");
  Serial.println("SENSORS ON -> COLLECTING 3000 SAMPLES (15s)");
  Serial.println("==========================================");

  unsigned long startTime = millis();

  while (sampleCount < TOTAL_SAMPLES) {
    if (checkSerialStop() || bleStopRequested) {
      bleStopRequested = false;
      Serial.println("STATUS,STOPPED");
      if (bleConnected && pTxChar) {
        pTxChar->setValue((uint8_t *)"STATUS,STOPPED\n", 15);
        pTxChar->notify();
      }
      Serial.println("ACQUISITION CANCELLED BY USER");
      setSensorsPower(false);
      return false;
    }

    // Update toe sensor FIFO
    uint16_t toeIR;
    uint16_t toeRED;

    if (readMAX30100(toeIR, toeRED)) {
      lastToeIR = toeIR;
      lastToeRED = toeRED;
    }

    // Update finger sensor
    max30102.check();

    // Read IMU once per batch (outside inner loop) to avoid I2C bus congestion
    float ax = 0, ay = 0, az = 0;
    float gx = 0, gy = 0, gz = 0;
    readMPU6500(ax, ay, az, gx, gy, gz);

    while (max30102.available() && sampleCount < TOTAL_SAMPLES) {
      uint32_t fingerIR = max30102.getFIFOIR();
      uint32_t fingerRED = max30102.getFIFORed();

      uint32_t timestamp = micros();

      char row[128];
      snprintf(row, sizeof(row),
               "%d,%lu,%lu,%lu,%u,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
               sampleCount, (unsigned long)timestamp, (unsigned long)fingerIR,
               (unsigned long)fingerRED, (unsigned)lastToeIR,
               (unsigned)lastToeRED, ax, ay, az, gx, gy, gz);

      csvData += row;

      // Stream live telemetry for Web Serial chart (DATA,ts_ms,finger,toe,gx,gy,gz)
      Serial.printf("DATA,%lu,%lu,%lu,%.1f,%.1f,%.1f\n",
                    (unsigned long)(timestamp / 1000), (unsigned long)fingerIR,
                    (unsigned long)lastToeIR, gx, gy, gz);

      // Stream live to Web Bluetooth if connected
      if (bleConnected && pTxChar && sampleCount % 2 == 0) {
        char bleBuf[48];
        snprintf(bleBuf, sizeof(bleBuf), "DATA,%lu,%lu,%lu\n",
                 (unsigned long)(timestamp / 1000), (unsigned long)fingerIR,
                 (unsigned long)lastToeIR);
        pTxChar->setValue((uint8_t *)bleBuf, strlen(bleBuf));
        pTxChar->notify();
      }

      sampleCount++;

      max30102.nextSample();

      if (sampleCount % 200 == 0) {
        Serial.printf("Progress: %d/3000\n", sampleCount);
      }
    }

    // Safety timeout (25 seconds for a 15-second window)
    if (millis() - startTime > 25000) {
      Serial.println("ERROR: Acquisition timeout");
      Serial.println("STATUS,STOPPED");
      setSensorsPower(false);
      return false;
    }

    if (!max30102.available()) {
      delay(1);
    }
  }

  // Acquisition finished: turn sensors OFF immediately!
  setSensorsPower(false);

  // Send completion message so dashboard immediately stops timer and runs model
  Serial.println("STATUS,DONE");
  if (bleConnected && pTxChar) {
    pTxChar->setValue((uint8_t *)"STATUS,DONE\n", 12);
    pTxChar->notify();
  }

  Serial.println();
  Serial.printf("3000 SAMPLES COMPLETED in %lu ms -> SENSORS TURNED OFF\n", millis() - startTime);

  Serial.print("CSV SIZE: ");
  Serial.print(csvData.length());
  Serial.println(" bytes");

  return true;
}

// =====================================================
// SETUP
// =====================================================

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10); // Prevent blocking on partial serial lines

  delay(2000);

  Serial.println();
  Serial.println("==========================================");
  Serial.println("PVD PPG SMART PATCH");
  Serial.println("ESP32-S3");
  Serial.println("==========================================");

  // -------------------------------------------------
  // MAIN I2C BUS
  // MAX30102 + MPU6500
  // -------------------------------------------------

  Wire.begin(SDA_PIN, SCL_PIN, 400000);
  delay(100);

  // -------------------------------------------------
  // MAX30102
  // -------------------------------------------------

  Serial.println("Initializing MAX30102...");

  if (!max30102.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 FAILED");
    while (1)
      delay(1000);
  }

  max30102.setup(60, 1, 2, 200, 411, 4096);
  max30102.setPulseAmplitudeGreen(0);
  Serial.println("MAX30102 INITIALIZED");

  // -------------------------------------------------
  // MPU6500
  // -------------------------------------------------

  if (!initMPU6500()) {
    Serial.println("MPU6500 FAILED");
    while (1)
      delay(1000);
  }
  Serial.println("MPU6500 INITIALIZED");

  // -------------------------------------------------
  // MAX30100
  // -------------------------------------------------

  if (!initMAX30100()) {
    Serial.println("MAX30100 FAILED");
    while (1)
      delay(1000);
  }

  // -------------------------------------------------
  // WIFI
  // -------------------------------------------------

  connectWiFi();

  // -------------------------------------------------
  // BLUETOOTH BLE
  // -------------------------------------------------

  initBLE();

  // -------------------------------------------------
  // SENSORS OFF BY DEFAULT (STANDBY)
  // -------------------------------------------------
  setSensorsPower(false);

  Serial.println();
  Serial.println("==========================================");
  Serial.println("ALL SENSORS READY & CURRENTLY OFF");
  Serial.println("Sensors will turn ON only when 'Start Rec' is clicked!");
  Serial.println("==========================================");
}

// =====================================================
// LOOP (STANDBY & TRIGGER LISTENER)
// =====================================================

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  bool startTriggered = false;
  bool isCloudTrigger = false;

  // 1. Check Serial command (Web Serial or USB Monitor)
  if (checkSerialStart()) {
    Serial.println("\n[TRIGGER] Start command received via Serial!");
    startTriggered = true;
  }

  // 2. Check Bluetooth BLE command
  if (!startTriggered && bleStartRequested) {
    bleStartRequested = false;
    Serial.println("\n[TRIGGER] Start command received via Bluetooth BLE!");
    startTriggered = true;
  }

  // 3. Check Cloud Server status (polling only when idle and not in active BLE/Serial session)
  static unsigned long lastPoll = 0;
  if (!startTriggered && !bleConnected && millis() - lastPoll > 4000) {
    lastPoll = millis();
    if (checkServerStart()) {
      Serial.println("\n[TRIGGER] 'Start Rec' clicked on Web Dashboard via WiFi!");
      startTriggered = true;
      isCloudTrigger = true;
    }
  }

  if (startTriggered) {
    bool acquired = acquirePPGWindow();

    if (acquired) {
      if (isCloudTrigger) {
        bool uploaded = sendDataToDashboard();
        if (uploaded) {
          Serial.println("CYCLE COMPLETE");
        } else {
          Serial.println("UPLOAD FAILED");
        }
      } else {
        Serial.println("CYCLE COMPLETE (Streamed to Dashboard via Serial/BLE)");
      }
    } else {
      Serial.println("ACQUISITION ABORTED / FAILED");
    }

    // Ensure sensors stay OFF in standby
    setSensorsPower(false);
    Serial.println(
        "\n[STANDBY] Sensors are OFF. Waiting for next 'Start Rec'...");
  }

  delay(20);
}