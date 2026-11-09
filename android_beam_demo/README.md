# CLARIS beam search demo

Single-screen Android app that runs the S2U pipeline on bundled wTIMIT recordings and visualises the beam search.

wav → 80-bin fbank + CMVN (kaldi-native-fbank) → encoder (LiteRT, int8 weights) → beam-search decoder (LiteRT-LM bridge,
int8 weights, width 3) → units → duration predictor (LiteRT) → vocoder (ONNX Runtime, fp32) → 16 kHz audio.

## Build
Use a local Gradle 8.9 (AGP 8.7.3); from this folder:

    gradle :app:assembleDebug

Needs the Android SDK and NDK (CMake 3.22.1). `local.properties` specifies `sdk.dir=<path_to_android_sdk>` (configured automatically by Android Studio). Configuring needs no
network (kissfft is vendored in `app/src/main/cpp/third_party`).

## Layout
- `app/src/main/cpp/` – JNI (`native-lib.cpp`), engine (`claris_inference_engine.*`), fbank front end.
- `app/src/main/jniLibs/arm64-v8a/liblitert_lm_shared.so` – prebuilt bridge. Its source is **not** in this repo
  (built with Bazel from the LiteRT-LM checkout, `runtime/engine/litert_lm_shared.cc`). It writes the search tree to
  `beam_trajectory.txt` in the cache dir, which the UI draws.
- `app/src/main/assets/models/` – encoder, decoder (`decoder_lm_beam3.litertlm`), duration predictor, vocoder.
- `app/src/main/assets/samples/` – 12 wTIMIT validation clips + `samples.tsv` (reference text), from `backend/claris_retrieval/samples`.

## Notes
- Beam width is fixed at 3: the decoder graph's batch dimension is 3. Another width needs a re-export
  (`backend/s2u_inference/export_litert_lm.py`) and a matching bridge build.
- Everything runs on CPU. arm64-v8a only, minSdk 31.
