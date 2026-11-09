#ifndef CLARIS_INFERENCE_ENGINE_H
#define CLARIS_INFERENCE_ENGINE_H

#include <memory>
#include <string>
#include <vector>
#include <android/asset_manager.h>
#include "onnxruntime_cxx_api.h"

// Status codes (the engine only needs ok / failure; no LiteRT headers required).
typedef int LiteRtStatus;
constexpr LiteRtStatus kLiteRtStatusOk = 0;
constexpr LiteRtStatus kLiteRtStatusErrorRuntimeFailure = 1;

// Bridge functions from liblitert_lm_shared.so
extern "C" {
    typedef void* LiteRtLmEnginePtr;
    typedef void* LiteRtEncoderEnginePtr;

    LiteRtLmEnginePtr LiteRtCreateLmEngineSimple(const char* model_path, int backend_code, int num_threads, const char* cache_dir);
    int LiteRtRunClarisInference(LiteRtLmEnginePtr engine_ptr, const float* spectrogram, size_t spectrogram_size, const int32_t* enc_mask, int32_t* out_tokens, int max_tokens, int sampling_method, int top_k, float top_p, float temperature, float* prefill_ms, float* decode_ms, bool enable_debug_logging, const char* cache_dir);
    int LiteRtRunClarisBeamSearch(LiteRtLmEnginePtr engine_ptr, const float* spectrogram, size_t spectrogram_size, const int32_t* enc_mask, int32_t* out_tokens, int max_tokens, int beam_size, float* prefill_ms, float* decode_ms, bool enable_debug_logging, const char* cache_dir);

    LiteRtEncoderEnginePtr LiteRtCreateEncoderEngine(const char* model_path, int backend_code);
    int LiteRtRunEncoder(LiteRtEncoderEnginePtr engine_ptr, const float* mel_spectrogram, size_t mel_frames, float* out_enc, size_t max_enc_frames, int32_t* out_mask, size_t max_mask_frames, int* actual_frames);
    void LiteRtDestroyEncoderEngine(LiteRtEncoderEnginePtr engine_ptr);

    typedef void* LiteRtVocoderEnginePtr;
    LiteRtVocoderEnginePtr LiteRtCreateVocoderEngine(const char* dur_model_path, const char* vocoder_model_path, const char* embeddings_path, int backend_code);
    void LiteRtDestroyVocoderEngine(LiteRtVocoderEnginePtr engine_ptr);
    int LiteRtRunVocoder(LiteRtVocoderEnginePtr engine_ptr, const int32_t* units, size_t num_units, float* out_audio, size_t max_audio_samples, int* actual_samples);
    int LiteRtRunDurationPredictorOnly(LiteRtVocoderEnginePtr engine_ptr, const int32_t* units, size_t num_units, int32_t* out_repeated, size_t max_repeated);

    int LiteRtGetLastDecoderLogits(float* out_logits, size_t max_size);
}

struct InferenceResult {
    std::vector<int32_t> tokens;
    std::vector<float> encoderOutput; // Shape: [T, 256]
    std::vector<float> decoderLogits; // Shape: [1004] (first pass raw logits)
    float prefill_ms = 0.0f;
    float decode_ms = 0.0f;
    float total_ms = 0.0f;
    float encoder_ms = 0.0f;
};

class ClarisInferenceEngine {
public:
    ClarisInferenceEngine();
    ~ClarisInferenceEngine();

    LiteRtStatus Initialize(AAssetManager* assetManager, 
                           const std::string& encoderPath, 
                           const std::string& decoderPath,
                           const std::string& cacheDir,
                           int encoderDevice, int decoderDevice);

    LiteRtStatus InitializeVocoder(AAssetManager* assetManager,
                                   const std::string& durPath,
                                   const std::string& vocoderPath,
                                   const std::string& embeddingsPath,
                                   const std::string& cacheDir,
                                   int vocoderDevice);

    InferenceResult RunInference(const std::vector<float>& melSpectrogram, int samplingMethod = 0, int top_k = 10, float top_p = 0.9f, float temperature = 0.85f, int beam_size = 3, bool enable_debug_logging = false);
    
    std::vector<float> ExtractMelSpec(const float* waveform, size_t size, int sampleRate);

    std::vector<float> RunVocoder(const std::vector<int>& units);

private:
    LiteRtLmEnginePtr decoderEngine_ = nullptr;
    LiteRtEncoderEnginePtr encoderEngine_ = nullptr;
    LiteRtVocoderEnginePtr vocoderEngine_ = nullptr;
    std::string encoderPath_;
    std::string decoderPath_;
    std::string cacheDir_;

    bool InitializeEncoder(AAssetManager* assetManager, const std::string& path, const std::string& cacheDir, int encoderDevice);
    bool InitializeDecoder(AAssetManager* assetManager, const std::string& path, const std::string& cacheDir, int decoderDevice);

    std::vector<float> persistentEncoderOutput_;
    std::vector<int32_t> persistentEncoderMask_;

    std::unique_ptr<Ort::Env> ortEnv_;
    std::unique_ptr<Ort::Session> ortVocoderSession_;
};

#endif // CLARIS_INFERENCE_ENGINE_H
