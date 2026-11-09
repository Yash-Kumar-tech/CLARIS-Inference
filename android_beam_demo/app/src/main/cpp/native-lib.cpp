// JNI entry points for the CLARIS beam-search demo: wav -> mel -> encoder -> beam-search decoder -> units,
// then units -> duration predictor -> vocoder -> waveform.
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "claris_inference_engine.h"

namespace {

std::unique_ptr<ClarisInferenceEngine> g_engine;
std::vector<int32_t> g_lastUnits;
std::mutex g_mutex;  // engine buffers are reused; calls must not overlap

std::string Json(bool ok, const std::string& error, long long initMs, float melMs, const InferenceResult& r,
                 const std::vector<int32_t>& units) {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "{\"ok\":%s,\"error\":\"%s\",\"initMs\":%lld,\"melMs\":%.1f,\"encoderMs\":%.1f,"
                "\"prefillMs\":%.1f,\"decodeMs\":%.1f,\"totalMs\":%.1f,\"numUnits\":%zu}",
                ok ? "true" : "false", error.c_str(), initMs, melMs, r.encoder_ms, r.prefill_ms, r.decode_ms,
                r.total_ms, units.size());
  return buf;
}

}  // namespace

// Runs mel + encoder + beam search. Returns a JSON timing summary; the units are read with getLastUnits().
extern "C" JNIEXPORT jstring JNICALL
Java_com_yashtech_clarisbeam_NativeInference_runBeamSearch(JNIEnv* env, jobject, jobject assetManager_j,
                                                           jstring cacheDir_j, jfloatArray waveform_j,
                                                           jint sampleRate, jint beamSize) {
  std::lock_guard<std::mutex> lock(g_mutex);
  AAssetManager* assetManager = AAssetManager_fromJava(env, assetManager_j);
  const char* c = env->GetStringUTFChars(cacheDir_j, nullptr);
  std::string cacheDir(c);
  env->ReleaseStringUTFChars(cacheDir_j, c);
  // The bridge writes beam_trajectory.txt (the search tree) into this directory.
  setenv("CLARIS_CACHE_DIR", cacheDir.c_str(), 1);

  jsize n = env->GetArrayLength(waveform_j);
  std::vector<float> waveform(n);
  env->GetFloatArrayRegion(waveform_j, 0, n, waveform.data());
  for (float& s : waveform) s *= 32768.0f;  // fbank expects 16-bit scale, as in training

  g_lastUnits.clear();
  InferenceResult empty;
  auto t0 = std::chrono::steady_clock::now();
  if (!g_engine) g_engine = std::make_unique<ClarisInferenceEngine>();
  LiteRtStatus st = g_engine->Initialize(assetManager, "models/encoder.litert", "models/decoder_lm_beam3.litertlm",
                                         cacheDir, /*encoderDevice=*/0, /*decoderDevice=*/0);
  long long initMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
  if (st != kLiteRtStatusOk) {
    g_engine.reset();
    return env->NewStringUTF(Json(false, "engine initialization failed", initMs, 0, empty, {}).c_str());
  }

  auto t1 = std::chrono::steady_clock::now();
  std::vector<float> mel = g_engine->ExtractMelSpec(waveform.data(), waveform.size(), sampleRate);
  float melMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t1).count();

  InferenceResult res = g_engine->RunInference(mel, /*samplingMethod=*/1, 10, 0.9f, 0.85f, beamSize, false);
  if (res.tokens.empty()) return env->NewStringUTF(Json(false, "decoder returned no units", initMs, melMs, res, {}).c_str());

  for (int32_t t : res.tokens)
    if (t >= 4) g_lastUnits.push_back(t - 4);  // drop specials; unit u is stored as u + 4
  return env->NewStringUTF(Json(true, "", initMs, melMs, res, g_lastUnits).c_str());
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_yashtech_clarisbeam_NativeInference_getLastUnits(JNIEnv* env, jobject) {
  std::lock_guard<std::mutex> lock(g_mutex);
  jintArray out = env->NewIntArray((jsize)g_lastUnits.size());
  env->SetIntArrayRegion(out, 0, (jsize)g_lastUnits.size(), g_lastUnits.data());
  return out;
}

// Duration predictor + vocoder. Returns 16 kHz mono samples in [-1, 1] (empty on failure).
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_yashtech_clarisbeam_NativeInference_synthesize(JNIEnv* env, jobject, jobject assetManager_j,
                                                        jstring cacheDir_j, jintArray units_j) {
  std::lock_guard<std::mutex> lock(g_mutex);
  AAssetManager* assetManager = AAssetManager_fromJava(env, assetManager_j);
  const char* c = env->GetStringUTFChars(cacheDir_j, nullptr);
  std::string cacheDir(c);
  env->ReleaseStringUTFChars(cacheDir_j, c);

  jsize n = env->GetArrayLength(units_j);
  std::vector<int> units(n);
  env->GetIntArrayRegion(units_j, 0, n, units.data());

  if (!g_engine) g_engine = std::make_unique<ClarisInferenceEngine>();
  if (g_engine->InitializeVocoder(assetManager, "models/dur_predictor.litert", "models/vocoder.onnx",
                                  "models/vocoder_embeddings.bin", cacheDir, /*vocoderDevice=*/0) != kLiteRtStatusOk)
    return env->NewFloatArray(0);

  std::vector<float> audio = g_engine->RunVocoder(units);
  jfloatArray out = env->NewFloatArray((jsize)audio.size());
  env->SetFloatArrayRegion(out, 0, (jsize)audio.size(), audio.data());
  return out;
}
