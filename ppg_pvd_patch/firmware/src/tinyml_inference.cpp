#include "tinyml_inference.h"
#include <Arduino.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "../model/model_data.h"

namespace {
constexpr int kArenaSize =
    8 * 1024; // MLP is small; 8KB arena is generous headroom
uint8_t tensorArena[kArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
} // namespace

bool TinyMlInference::begin() {
  model = tflite::GetModel(g_pvd_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[TinyML] Model schema version mismatch");
    modelLoaded_ = false;
    return false;
  }

  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensorArena, kArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[TinyML] AllocateTensors failed");
    modelLoaded_ = false;
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  modelLoaded_ = true;
  return true;
}

RiskLevel TinyMlInference::predict(const float *features, float *probsOut) {
  if (!modelLoaded_)
    return predictHeuristic(features);

  for (int i = 0; i < FEATURE_VEC_LEN; ++i) {
    input->data.f[i] = features[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[TinyML] Invoke failed, falling back to heuristic");
    return predictHeuristic(features);
  }

  int bestClass = 0;
  float bestProb = output->data.f[0];
  for (int i = 1; i < 3; ++i) {
    if (probsOut)
      probsOut[i] = output->data.f[i];
    if (output->data.f[i] > bestProb) {
      bestProb = output->data.f[i];
      bestClass = i;
    }
  }
  if (probsOut)
    probsOut[0] = output->data.f[0];

  return static_cast<RiskLevel>(bestClass);
}

RiskLevel TinyMlInference::predictHeuristic(const float *features) {
  // Backstop rule-based classifier using PI_ratio (idx 2) and PTT_ft (idx 3),
  // mirrors the thresholds in config.h. Only used if the TFLite model is
  // unavailable — keeps the device from failing silently.
  float pi_ratio = features[2];
  float ptt_ft = features[3];

  if (pi_ratio < PI_RATIO_HIGH_THRESH || ptt_ft > PTT_FT_HIGH_MS) {
    return ::RiskLevel::HIGH_RISK;
  }
  if (pi_ratio < PI_RATIO_MODERATE_THRESH || ptt_ft > PTT_FT_MODERATE_MS) {
    return ::RiskLevel::MODERATE;
  }
  return ::RiskLevel::NORMAL;
}
