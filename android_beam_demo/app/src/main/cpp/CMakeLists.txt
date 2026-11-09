cmake_minimum_required(VERSION 3.22.1)
project("claris-beam")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -ftree-vectorize")

add_library(native-inference SHARED
        native-lib.cpp
        claris_inference_engine.cpp
        audio_utils.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kaldi-native-fbank)

set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
# kissfft is vendored so configuring needs no network access
set(FETCHCONTENT_SOURCE_DIR_KISSFFT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kissfft CACHE PATH "" FORCE)
add_subdirectory(kaldi-native-fbank)

find_library(log-lib log)

# Prebuilt libraries (arm64-v8a). liblitert_lm_shared.so is the LiteRT / LiteRT-LM bridge (encoder engine,
# decoder engine with batched beam search, duration predictor); the GPU/sampler libs next to it are dlopen'd by it.
set(JNI_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})
add_library(litert-lm-lib SHARED IMPORTED)
set_target_properties(litert-lm-lib PROPERTIES IMPORTED_LOCATION ${JNI_LIBS}/liblitert_lm_shared.so)
add_library(onnxruntime-lib SHARED IMPORTED)
set_target_properties(onnxruntime-lib PROPERTIES IMPORTED_LOCATION ${JNI_LIBS}/libonnxruntime.so)

target_link_libraries(native-inference ${log-lib} litert-lm-lib onnxruntime-lib kaldi-native-fbank-core android)
