#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "MAX30105.h"

// =====================================================
// DEVICE CONFIGURATION
// =====================================================

#define DEVICE_ID "PVD_PATCH_001"

// ESP32-S3 I2C pins
#define SDA_PIN 8
#define SCL_PIN 9

// PPG configuration
#define SAMPLE_RATE 200
#define WINDOW_SECONDS 8
#define NUM_SAMPLES (SAMPLE_RATE * WINDOW_SECONDS)

// =====================================================
// WIFI CONFIGURATION
// =====================================================

const char* WIFI_SSID = "Ranjith's S25FE";
const char* WIFI_PASSWORD = "12345678";

// =====================================================
// DASHBOARD API
// =====================================================

// CHANGE THIS TO YOUR DASHBOARD BACKEND API
const char* SERVER_URL =
    "http://10.122.243.56:8001/api/ppg";

// =====================================================
// MAX30102
// =====================================================

MAX30105 particleSensor;

// =====================================================
// DATA BUFFER
// =====================================================

struct PPGSample
{
    unsigned long timestamp;
    long red;
    long ir;
};

PPGSample samples[NUM_SAMPLES];

int sampleIndex = 0;

unsigned long lastSampleTime = 0;

const unsigned long sampleInterval =
    1000000UL / SAMPLE_RATE;

// =====================================================
// WIFI CONNECTION
// =====================================================

void connectWiFi()
{
    Serial.println();
    Serial.println("Connecting to Wi-Fi...");

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;

    while (WiFi.status() != WL_CONNECTED && attempts < 30)
    {
        delay(500);

        Serial.print(".");

        attempts++;
    }

    Serial.println();

    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.println("Wi-Fi connected!");

        Serial.print("ESP32 IP: ");
        Serial.println(WiFi.localIP());

        Serial.print("RSSI: ");
        Serial.println(WiFi.RSSI());
    }
    else
    {
        Serial.println("Wi-Fi connection FAILED.");
    }
}

// =====================================================
// MAX30102 INITIALIZATION
// =====================================================

bool initializePPG()
{
    Serial.println("Initializing MAX30102...");

    Wire.begin(SDA_PIN, SCL_PIN);

    if (!particleSensor.begin(Wire, I2C_SPEED_FAST))
    {
        Serial.println("ERROR: MAX30102 not detected!");

        return false;
    }

    Serial.println("MAX30102 detected.");

    // MAX30102 configuration

    byte ledBrightness = 60;

    byte sampleAverage = 1;

    byte ledMode = 2;     // RED + IR

    int sampleRate = 200;

    int pulseWidth = 411;

    int adcRange = 4096;

    particleSensor.setup(
        ledBrightness,
        sampleAverage,
        ledMode,
        sampleRate,
        pulseWidth,
        adcRange
    );

    particleSensor.setPulseAmplitudeRed(0x24);

    particleSensor.setPulseAmplitudeIR(0x24);

    Serial.println("PPG configured.");

    Serial.println("Sample rate: 200 Hz");

    return true;
}

// =====================================================
// SEND DATA TO DASHBOARD
// =====================================================

bool sendPPGData()
{
    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("Wi-Fi disconnected.");

        connectWiFi();

        if (WiFi.status() != WL_CONNECTED)
        {
            return false;
        }
    }

    Serial.println();
    Serial.println("Preparing PPG data...");

    // -------------------------------------------------
    // Create CSV payload
    // -------------------------------------------------

    String payload;

    payload.reserve(60000);

    payload += "device_id,timestamp_ms,red,ir\n";

    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        payload += DEVICE_ID;
        payload += ",";

        payload += String(samples[i].timestamp);
        payload += ",";

        payload += String(samples[i].red);
        payload += ",";

        payload += String(samples[i].ir);

        payload += "\n";
    }

    Serial.print("Payload size: ");

    Serial.print(payload.length());

    Serial.println(" bytes");

    // -------------------------------------------------
    // HTTP POST
    // -------------------------------------------------

    HTTPClient http;

    Serial.println("Sending data to dashboard...");

    http.begin(SERVER_URL);

    http.addHeader(
        "Content-Type",
        "text/csv"
    );

    http.addHeader(
        "X-Device-ID",
        DEVICE_ID
    );

    int httpCode = http.POST(payload);

    Serial.print("HTTP response: ");

    Serial.println(httpCode);

    if (httpCode > 0)
    {
        String response = http.getString();

        Serial.println("Server response:");

        Serial.println(response);

        http.end();

        return httpCode >= 200 && httpCode < 300;
    }

    Serial.print("HTTP error: ");

    Serial.println(
        http.errorToString(httpCode)
    );

    http.end();

    return false;
}

// =====================================================
// COLLECT PPG WINDOW
// =====================================================

void collectPPGWindow()
{
    Serial.println();
    Serial.println("==============================");

    Serial.println(
        "Collecting 8-second PPG window"
    );

    Serial.println("==============================");

    sampleIndex = 0;

    lastSampleTime = micros();

    while (sampleIndex < NUM_SAMPLES)
    {
        unsigned long currentTime = micros();

        if (currentTime - lastSampleTime >= sampleInterval)
        {
            lastSampleTime += sampleInterval;

            long redValue =
                particleSensor.getRed();

            long irValue =
                particleSensor.getIR();

            samples[sampleIndex].timestamp =
                millis();

            samples[sampleIndex].red =
                redValue;

            samples[sampleIndex].ir =
                irValue;

            sampleIndex++;

            particleSensor.nextSample();

            // Progress indication
            if (sampleIndex % 200 == 0)
            {
                Serial.print("Samples: ");

                Serial.print(sampleIndex);

                Serial.print("/");

                Serial.println(NUM_SAMPLES);
            }
        }
    }

    Serial.println();

    Serial.println(
        "PPG window complete."
    );
}

// =====================================================
// SETUP
// =====================================================

void setup()
{
    Serial.begin(115200);

    delay(1500);

    Serial.println();
    Serial.println(
        "========================================"
    );

    Serial.println(
        " PPG-BASED PVD EARLY WARNING PATCH"
    );

    Serial.println(
        " ESP32-S3 DATA ACQUISITION SYSTEM"
    );

    Serial.println(
        "========================================"
    );

    Serial.print("Device ID: ");

    Serial.println(DEVICE_ID);

    Serial.print("Sampling rate: ");

    Serial.print(SAMPLE_RATE);

    Serial.println(" Hz");

    Serial.print("Window: ");

    Serial.print(WINDOW_SECONDS);

    Serial.println(" seconds");

    Serial.print("Samples/window: ");

    Serial.println(NUM_SAMPLES);

    // Initialize PPG

    if (!initializePPG())
    {
        Serial.println(
            "PPG initialization failed."
        );

        while (1)
        {
            delay(1000);
        }
    }

    // Connect Wi-Fi

    connectWiFi();

    Serial.println();

    Serial.println(
        "SYSTEM READY"
    );
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop()
{
    // Collect 8-second PPG window

    collectPPGWindow();

    // Send to dashboard

    bool success = sendPPGData();

    if (success)
    {
        Serial.println();

        Serial.println(
            "PPG DATA SENT SUCCESSFULLY"
        );
    }
    else
    {
        Serial.println();

        Serial.println(
            "PPG DATA SEND FAILED"
        );
    }

    Serial.println();

    Serial.println(
        "Starting next acquisition..."
    );

    delay(1000);
}
