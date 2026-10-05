#include <Arduino.h>
#include "config.h"
#include "ppg_sensor.h"
#include "signal_processing.h"
#include "feature_extraction.h"
#include "tinyml_inference.h"
#include "ble_service.h"

// ---------- Globals shared across tasks ----------
static PpgSensor ppgSensor;
static BleService bleService;
static TinyMlInference tinyMl;

static QueueHandle_t sampleQueue;       // sampling task -> processing task
static SemaphoreHandle_t bufferMutex;

static SiteBuffer fingerBuf;
static SiteBuffer toeBuf;

// ---------- Task: Sampling (Core 0, highest priority, hard real-time) ----------
void samplingTask(void *pv) {
    const TickType_t period = pdMS_TO_TICKS(1000 / PPG_SAMPLE_RATE_HZ);
    TickType_t lastWake = xTaskGetTickCount();

    for (;;) {
        PpgSample s = ppgSensor.readSample();
        xQueueSend(sampleQueue, &s, 0);  // non-blocking; drop on overflow
        vTaskDelayUntil(&lastWake, period);
    }
}

// ---------- Task: DSP + buffering (Core 1) ----------
void processingTask(void *pv) {
    BiquadFilter filtFinger = makeBandpassFilter();
    BiquadFilter filtToe = makeBandpassFilter();

    PpgSample s;
    for (;;) {
        if (xQueueReceive(sampleQueue, &s, portMAX_DELAY) == pdTRUE) {
            float rawFinger = static_cast<float>(s.ir_finger);
            float rawToe = static_cast<float>(s.ir_toe);

            float filteredFinger = filtFinger.process(rawFinger);
            float filteredToe = filtToe.process(rawToe);

            if (xSemaphoreTake(bufferMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                fingerBuf.raw.push_back(rawFinger);
                fingerBuf.filtered.push_back(filteredFinger);
                fingerBuf.timestamps_ms.push_back(s.timestamp_ms);

                toeBuf.raw.push_back(rawToe);
                toeBuf.filtered.push_back(filteredToe);
                toeBuf.timestamps_ms.push_back(s.timestamp_ms);

                // Keep a rolling window of PPG_BUFFER_LEN samples (~8s)
                if (fingerBuf.raw.size() > PPG_BUFFER_LEN) {
                    fingerBuf.raw.erase(fingerBuf.raw.begin());
                    fingerBuf.filtered.erase(fingerBuf.filtered.begin());
                    fingerBuf.timestamps_ms.erase(fingerBuf.timestamps_ms.begin());
                    toeBuf.raw.erase(toeBuf.raw.begin());
                    toeBuf.filtered.erase(toeBuf.filtered.begin());
                    toeBuf.timestamps_ms.erase(toeBuf.timestamps_ms.begin());
                }
                xSemaphoreGive(bufferMutex);
            }
        }
    }
}

// ---------- Task: Feature extraction + inference (Core 1, lower priority) ----------
void inferenceTask(void *pv) {
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(INFERENCE_PERIOD_MS));

        if (xSemaphoreTake(bufferMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            // DC mean for perfusion index
            auto meanOf = [](const std::vector<float> &v) {
                if (v.empty()) return 0.0f;
                float s = 0;
                for (float x : v) s += x;
                return s / v.size();
            };
            fingerBuf.dc_mean = meanOf(fingerBuf.raw);
            toeBuf.dc_mean = meanOf(toeBuf.raw);

            fingerBuf.peaks = detectPeaks(fingerBuf.filtered, fingerBuf.timestamps_ms);
            toeBuf.peaks = detectPeaks(toeBuf.filtered, toeBuf.timestamps_ms);

            float features[FEATURE_VEC_LEN];
            extractFeatures(fingerBuf, toeBuf, features);

            xSemaphoreGive(bufferMutex);

            float probs[3] = {0, 0, 0};
            RiskLevel risk = tinyMl.predict(features, probs);

            bleService.notifyFeatures(features, FEATURE_VEC_LEN);
            bleService.notifyRisk(risk);

            Serial.printf(
                "[Inference] PI_f=%.2f PI_t=%.2f ratio=%.2f PTT=%.1fms AI_f=%.1f AI_t=%.1f "
                "HRV=%.1f dicR=%.2f -> risk=%d (p=[%.2f,%.2f,%.2f])\n",
                features[0], features[1], features[2], features[3], features[4],
                features[5], features[6], features[7], static_cast<int>(risk),
                probs[0], probs[1], probs[2]);
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("PVD Early Warning Patch - booting");

    if (!ppgSensor.begin()) {
        Serial.println("FATAL: PPG sensor init failed");
    }

    tinyMl.begin();     // falls back to heuristic classifier if this fails
    bleService.begin();

    sampleQueue = xQueueCreate(64, sizeof(PpgSample));
    bufferMutex = xSemaphoreCreateMutex();

    fingerBuf.raw.reserve(PPG_BUFFER_LEN);
    fingerBuf.filtered.reserve(PPG_BUFFER_LEN);
    toeBuf.raw.reserve(PPG_BUFFER_LEN);
    toeBuf.filtered.reserve(PPG_BUFFER_LEN);

    xTaskCreatePinnedToCore(samplingTask, "Sampling", 4096, nullptr, 3, nullptr, 0);
    xTaskCreatePinnedToCore(processingTask, "Processing", 8192, nullptr, 2, nullptr, 1);
    xTaskCreatePinnedToCore(inferenceTask, "Inference", 8192, nullptr, 1, nullptr, 1);
}

void loop() {
    // All work happens in FreeRTOS tasks; idle here.
    vTaskDelay(pdMS_TO_TICKS(1000));
}
