#include "claris_inference_engine.h"
#include "nnapi_provider_factory.h"
#include "audio_utils.h"
#include <algorithm>
#include <android/asset_manager.h>
#include <android/log.h>
#include <chrono>
#include <fstream>
#include <sys/stat.h>

//#define LOG_TAG "ClarisInference"
//#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
//#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {
void LiteRtDestroyLmEngine(LiteRtLmEnginePtr engine_ptr);
}

ClarisInferenceEngine::ClarisInferenceEngine() {
    persistentEncoderOutput_.resize(250 * 256, 0.0f);
    persistentEncoderMask_.resize(250, 0);
}

ClarisInferenceEngine::~ClarisInferenceEngine() {
  if (decoderEngine_) {
    LiteRtDestroyLmEngine(decoderEngine_);
    decoderEngine_ = nullptr;
  }
  if (encoderEngine_) {
      LiteRtDestroyEncoderEngine(encoderEngine_);
      encoderEngine_ = nullptr;
  }
  if (vocoderEngine_) {
    LiteRtDestroyVocoderEngine(vocoderEngine_);
    vocoderEngine_ = nullptr;
  }
}

LiteRtStatus ClarisInferenceEngine::Initialize(AAssetManager *assetManager,
                                               const std::string &encoderPath,
                                               const std::string &decoderPath,
                                               const std::string &cacheDir,
                                               int encoderDevice, int decoderDevice) {
  if (encoderEngine_ && decoderEngine_) {
      return kLiteRtStatusOk;
  }
//  LOGI("INIT: Starting full pipeline initialization...");

  this->encoderPath_ = encoderPath;
  this->decoderPath_ = decoderPath;
  this->cacheDir_ = cacheDir;

  if (!InitializeEncoder(assetManager, encoderPath, cacheDir, encoderDevice)) {
//    LOGE("INIT: Failed to initialize Encoder.");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  if (!InitializeDecoder(assetManager, decoderPath, cacheDir, decoderDevice)) {
//    LOGE("INIT: Failed to initialize Decoder.");
    return kLiteRtStatusErrorRuntimeFailure;
  }

//  LOGI("INIT: Full pipeline ready.");
  return kLiteRtStatusOk;
}

bool ClarisInferenceEngine::InitializeEncoder(AAssetManager *assetManager,
                                              const std::string &path,
                                              const std::string &cacheDir,
                                              int encoderDevice) {
//  LOGI("--------------------------------------------------");
//  LOGI("INIT [ENCODER]: Starting initialization for %s", path.c_str());

  std::string extractedPath = cacheDir + "/encoder.litert";
//  LOGI("INIT [ENCODER]: Extracting to %s...", extractedPath.c_str());

  AAsset *asset =
      AAssetManager_open(assetManager, path.c_str(), AASSET_MODE_BUFFER);
  if (!asset) {
//    LOGE("INIT [ENCODER] ERROR: Failed to open asset: %s", path.c_str());
    return false;
  }

  const void *buffer = AAsset_getBuffer(asset);
  size_t size = AAsset_getLength(asset);
  std::ofstream ofs(extractedPath, std::ios::binary);
  if (!ofs) {
//    LOGE("INIT [ENCODER] ERROR: Failed to open output file: %s",
//         extractedPath.c_str());
    AAsset_close(asset);
    return false;
  }
  ofs.write((const char *)buffer, size);
  ofs.close();
  AAsset_close(asset);
//  LOGI("INIT [ENCODER]: Extraction complete. Size: %zu bytes", size);
//
//  LOGI("INIT [ENCODER]: Calling LiteRtCreateEncoderEngine...");
  int backendCode = (encoderDevice == 1) ? 2 : 1; // 1 = CPU, 2 = GPU. Note: INT64 incompatibility might still cause fallback.
  encoderEngine_ = LiteRtCreateEncoderEngine(extractedPath.c_str(), backendCode); 
  if (!encoderEngine_) {
//    LOGE("INIT [ENCODER] ERROR: Failed to create encoder engine.");
    return false;
  }

//  LOGI("INIT [ENCODER] SUCCESS: Encoder initialized.");
  return true;
}

bool ClarisInferenceEngine::InitializeDecoder(AAssetManager *assetManager,
                                              const std::string &path,
                                              const std::string &cacheDir,
                                              int decoderDevice) {
//  LOGI("--------------------------------------------------");
//  LOGI("INIT [DECODER]: Starting initialization for %s", path.c_str());

  std::string extractedPath = cacheDir + "/decoder.litertlm";
//  LOGI("INIT [DECODER]: Extracting to %s...", extractedPath.c_str());

  AAsset *asset =
      AAssetManager_open(assetManager, path.c_str(), AASSET_MODE_BUFFER);
  if (!asset) {
//    LOGE("INIT [DECODER] ERROR: Failed to open asset: %s", path.c_str());
    return false;
  }

  const void *buffer = AAsset_getBuffer(asset);
  size_t size = AAsset_getLength(asset);
  std::ofstream ofs(extractedPath, std::ios::binary);
  if (!ofs) {
//    LOGE("INIT [DECODER] ERROR: Failed to open output file: %s",
//         extractedPath.c_str());
    AAsset_close(asset);
    return false;
  }
  ofs.write((const char *)buffer, size);
  ofs.close();
  AAsset_close(asset);
//  LOGI("INIT [DECODER]: Extraction complete. Size: %zu bytes", size);

  int backendCode = (decoderDevice == 1) ? 4 : 3; // 4 = GPU, 3 = CPU
  int numThreads = 4;

//  LOGI("INIT [DECODER]: Calling stable bridge LiteRtCreateLmEngineSimple...");
  decoderEngine_ = LiteRtCreateLmEngineSimple(
      extractedPath.c_str(), backendCode, numThreads, ":nocache");

  if (!decoderEngine_) {
//    LOGE("INIT [DECODER] ERROR: Stable bridge failed to create engine.");
    return false;
  }

//  LOGI("INIT [DECODER] SUCCESS: Engine initialized through bridge.");
  return true;
}

LiteRtStatus ClarisInferenceEngine::InitializeVocoder(AAssetManager* assetManager,
                               const std::string& durPath,
                               const std::string& vocoderPath,
                               const std::string& embeddingsPath,
                               const std::string& cacheDir,
                               int vocoderDevice) {
  if (vocoderEngine_ && ortVocoderSession_) {
      return kLiteRtStatusOk;
  }
  
//  LOGI("INIT [VOCODER]: Starting initialization...");
  cacheDir_ = cacheDir;

  auto extractAsset = [&](const std::string& assetPath, const std::string& destPath) {
    AAsset *asset = AAssetManager_open(assetManager, assetPath.c_str(), AASSET_MODE_BUFFER);
    if (!asset) {
//      LOGE("INIT [VOCODER] ERROR: Failed to open asset: %s", assetPath.c_str());
      return false;
    }
    const void *buffer = AAsset_getBuffer(asset);
    size_t size = AAsset_getLength(asset);
    std::ofstream ofs(destPath, std::ios::binary);
    if (!ofs) {
//      LOGE("INIT [VOCODER] ERROR: Failed to open output file: %s", destPath.c_str());
      AAsset_close(asset);
      return false;
    }
    ofs.write((const char *)buffer, size);
    ofs.close();
    AAsset_close(asset);
    return true;
  };

  std::string durExtracted = cacheDir + "/dur_predictor.litert";
  std::string vocoderExtracted = cacheDir + "/vocoder.onnx";
  std::string vocoderDataExtracted = cacheDir + "/vocoder.onnx.data";
  std::string embExtracted = cacheDir + "/vocoder_embeddings.bin";

  if (!extractAsset(durPath, durExtracted) ||
      !extractAsset(vocoderPath, vocoderExtracted) ||
      !extractAsset(vocoderPath + ".data", vocoderDataExtracted) ||
      !extractAsset(embeddingsPath, embExtracted)) {
    return kLiteRtStatusErrorRuntimeFailure;
  }

  int backendCode = (vocoderDevice == 1) ? 2 : 1; // 1 = CPU, 2 = GPU
  // Initialize Duration Predictor in LiteRT
  // We pass durExtracted for both paths to avoid a std::string(nullptr) crash in the bridge,
  // since we only use the Duration Predictor from the bridge now.
  vocoderEngine_ = LiteRtCreateVocoderEngine(durExtracted.c_str(), durExtracted.c_str(), embExtracted.c_str(), backendCode);
  if (!vocoderEngine_) {
//    LOGE("INIT [VOCODER] ERROR: Failed to create duration predictor engine.");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // Initialize ONNX Runtime for Vocoder
  try {
    ortEnv_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ClarisVocoder");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (vocoderDevice == 2) {
        // Try NNAPI for GPU/NPU
        try {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions, 0));
//            LOGI("INIT [VOCODER]: NNAPI execution provider appended.");
        } catch (const Ort::Exception& e) {
//            LOGE("INIT [VOCODER] WARNING: Failed to append NNAPI, falling back to CPU. %s", e.what());
        }
    } else {
        // Try XNNPACK for CPU
        try {
            std::unordered_map<std::string, std::string> xnnpack_options = {
                {"intra_op_num_threads", "4"}
            };
            sessionOptions.AppendExecutionProvider("XNNPACK", xnnpack_options);
//            LOGI("INIT [VOCODER]: XNNPACK execution provider appended.");
        } catch (const Ort::Exception& e) {
//            LOGE("INIT [VOCODER] WARNING: Failed to append XNNPACK, falling back to standard CPU. %s", e.what());
        }
    }
    
    ortVocoderSession_ = std::make_unique<Ort::Session>(*ortEnv_, vocoderExtracted.c_str(), sessionOptions);
  } catch (const Ort::Exception& e) {
//    LOGE("INIT [VOCODER] ERROR: ONNX Runtime initialization failed: %s", e.what());
    return kLiteRtStatusErrorRuntimeFailure;
  }

//  LOGI("INIT [VOCODER] SUCCESS: Vocoder initialized.");
  return kLiteRtStatusOk;
}

InferenceResult ClarisInferenceEngine::RunInference(const std::vector<float> &melSpectrogram,
                                    int samplingMethod, int top_k, float top_p, float temperature, int beam_size, bool enable_debug_logging) {
  InferenceResult result;
  if (!decoderEngine_) {
//    LOGE("EXEC ERROR: Decoder engine not initialized.");
    return result;
  }
  if (!encoderEngine_) {
//    LOGE("EXEC ERROR: Encoder engine not initialized.");
    return result;
  }
  if (melSpectrogram.empty()) {
//    LOGE("EXEC ERROR: melSpectrogram is empty. Cannot run inference.");
    return result;
  }

//  LOGI("EXEC: Starting full inference pass (samplingMethod=%d, top_k=%d, top_p=%.3f, temperature=%.3f, beam_size=%d)...",
//       samplingMethod, top_k, top_p, temperature, beam_size);

  auto total_start = std::chrono::high_resolution_clock::now();

  // 1. Run the Encoder
  // The encoder model always outputs 250 frames (1000/4) regardless of actual
  // mel length — it has a static output shape. We must always allocate 250 frames
  // and collect all of them, just like the Python reference does. The padding mask
  // communicates which frames are valid to the decoder.
  size_t mel_frames = melSpectrogram.size() / 80;
  const size_t ENC_OUTPUT_FRAMES = 250; // Model fixed output: 1000 mel / 4 = 250
  int actual_frames = 0;

  auto enc_start = std::chrono::high_resolution_clock::now();
  int enc_status =
      LiteRtRunEncoder(encoderEngine_, melSpectrogram.data(), mel_frames,
                       persistentEncoderOutput_.data(), ENC_OUTPUT_FRAMES,
                       persistentEncoderMask_.data(), ENC_OUTPUT_FRAMES, &actual_frames);
  auto enc_end = std::chrono::high_resolution_clock::now();
  result.encoder_ms = std::chrono::duration<float, std::milli>(enc_end - enc_start).count();

  int valid_enc_frames = mel_frames;
  valid_enc_frames = (valid_enc_frames + 1) / 2;
  valid_enc_frames = (valid_enc_frames + 1) / 2;
  actual_frames = valid_enc_frames;


  if (enc_status < 0) {
//    LOGE("EXEC ERROR: Encoder failed: %d", enc_status);
    return result;
  }

  // UI plotting currently doesn't strictly need a copy, but to satisfy the struct
  // we could assign it, though it adds copy overhead. Better to just leave it empty.
  // result.encoderOutput = encOut;

  if (enable_debug_logging) {
      std::string melPath = cacheDir_ + "/debug_mel.bin";
      std::ofstream melfs(melPath, std::ios::binary);
      if (melfs) {
          melfs.write((const char*)melSpectrogram.data(), melSpectrogram.size() * sizeof(float));
          melfs.close();
      }
      
      std::string outPath = cacheDir_ + "/debug_enc_out.bin";
      std::ofstream ofs(outPath, std::ios::binary);
      if (ofs) {
          ofs.write((const char*)persistentEncoderOutput_.data(), persistentEncoderOutput_.size() * sizeof(float));
          ofs.close();
      }
      
      std::string maskPath = cacheDir_ + "/debug_enc_mask.bin";
      std::ofstream mfs(maskPath, std::ios::binary);
      if (mfs) {
          mfs.write((const char*)persistentEncoderMask_.data(), persistentEncoderMask_.size() * sizeof(int32_t));
          mfs.close();
      }
  }

  // Dump logic removed to improve performance

  // 3. Run Decoder using the Encoder representation
  // The decoder needs to know how many encoder frames are genuinely valid
  // (from real mel data, not zero-padding).
//  LOGI("EXEC: valid_enc_frames=%d (mel_frames=%zu)", valid_enc_frames, mel_frames);
  int32_t output_tokens[250];
  int max_tokens = valid_enc_frames;

  int num_generated = 0;
  if (samplingMethod == 1) { // Beam Search
    num_generated = LiteRtRunClarisBeamSearch(
        decoderEngine_, persistentEncoderOutput_.data(), persistentEncoderOutput_.size(), persistentEncoderMask_.data(),
        output_tokens, max_tokens,
        beam_size, &result.prefill_ms, &result.decode_ms, enable_debug_logging, cacheDir_.c_str());
  } else {
    num_generated = LiteRtRunClarisInference(
        decoderEngine_, persistentEncoderOutput_.data(), persistentEncoderOutput_.size(), persistentEncoderMask_.data(),
        output_tokens, max_tokens,
        samplingMethod, top_k, top_p, temperature, &result.prefill_ms, &result.decode_ms, enable_debug_logging, cacheDir_.c_str());
//    LOGI("EXEC: Decoder used encOut of size %zu floats (%zu frames)",
//         encOut.size(), encOut.size() / 256);
  }

  // Dump logic removed to improve performance

  if (num_generated < 0) {
//    LOGE("EXEC ERROR: Inference bridge returned error: %d", num_generated);
    return result;
  }

  for (int i = 0; i < num_generated; ++i) {
    result.tokens.push_back(output_tokens[i]);
  }

  auto total_end = std::chrono::high_resolution_clock::now();
  result.total_ms =
      std::chrono::duration<float, std::milli>(total_end - total_start).count();

//  LOGI("EXEC SUCCESS: Generated %d units in %.2f ms (Prefill: %.2f ms, Decode: "
//       "%.2f ms)",
//       num_generated, result.total_ms, result.prefill_ms, result.decode_ms);

  return result;
}

std::vector<float>
ClarisInferenceEngine::ExtractMelSpec(const float* waveform, size_t size, int sampleRate) {
  return ComputeFbankCmvn(waveform, size, sampleRate);
}

std::vector<float> ClarisInferenceEngine::RunVocoder(const std::vector<int>& units) {
  if (!vocoderEngine_ || !ortVocoderSession_) {
//    LOGE("EXEC ERROR: Vocoder engine not initialized.");
    return {};
  }
  if (units.empty()) {
//    LOGE("EXEC ERROR: Input units empty.");
    return {};
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  const size_t MAX_REPEATED = 2048;
  std::vector<int32_t> repeated_units(MAX_REPEATED);
  int actual_repeated = LiteRtRunDurationPredictorOnly(vocoderEngine_, units.data(), units.size(), repeated_units.data(), MAX_REPEATED);
  
  if (actual_repeated <= 0) {
//    LOGE("EXEC ERROR: Duration Predictor failed or returned 0 length.");
    return {};
  }
  
  repeated_units.resize(actual_repeated);

  // Read embeddings manually since we bypassed C++ bridge vocoder
  // We need to look up embeddings for the repeated units to pass to ONNX
  std::string embExtracted = cacheDir_ + "/vocoder_embeddings.bin";
  std::ifstream ifs(embExtracted, std::ios::binary);
  if (!ifs) {
//    LOGE("EXEC ERROR: Failed to load embeddings.");
    return {};
  }
  std::vector<float> all_embeddings(1000 * 128);
  ifs.read((char*)all_embeddings.data(), all_embeddings.size() * sizeof(float));
  ifs.close();

  // Prepare input for ONNX: [1, 128, actual_repeated]
  std::vector<float> vocoder_in(128 * actual_repeated, 0.0f);
  for (size_t i = 0; i < actual_repeated; ++i) {
    int token = repeated_units[i];
    if (token >= 0 && token < 1000) {
      for (size_t d = 0; d < 128; ++d) {
        vocoder_in[d * actual_repeated + i] = all_embeddings[token * 128 + d];
      }
    }
  }

  try {
      Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      
      std::vector<int64_t> inputShape = {1, 128, static_cast<int64_t>(actual_repeated)};
      Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, vocoder_in.data(), vocoder_in.size(), inputShape.data(), inputShape.size());
      
      const char* inputNames[] = {"input"};
      const char* outputNames[] = {"output"};
      
      auto outputTensors = ortVocoderSession_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
      
      float* outData = outputTensors.front().GetTensorMutableData<float>();
      size_t outSize = outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
      
      std::vector<float> audio_out(outData, outData + outSize);

      auto t_end = std::chrono::high_resolution_clock::now();
      float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
//      LOGI("EXEC: ONNX Vocoder generated %zu samples in %.2f ms", outSize, ms);
      
      return audio_out;
      
  } catch (const Ort::Exception& e) {
//      LOGE("EXEC ERROR: ONNX Runtime Inference failed: %s", e.what());
      return {};
  }
}