package com.yashtech.clarisbeam

import android.content.res.AssetManager

/** JNI bridge to libnative-inference.so (see app/src/main/cpp/native-lib.cpp). */
object NativeInference {
    init {
        System.loadLibrary("litert_lm_shared")
        System.loadLibrary("native-inference")
    }

    /**
     * Mel features, encoder and beam-search decoder on a mono 16 kHz waveform in [-1, 1].
     * Returns a JSON timing summary; the predicted units come from [getLastUnits] and the search tree is written to
     * `beam_trajectory.txt` in [cacheDir].
     */
    external fun runBeamSearch(
        assetManager: AssetManager,
        cacheDir: String,
        waveform: FloatArray,
        sampleRate: Int,
        beamSize: Int,
    ): String

    external fun getLastUnits(): IntArray

    /** Duration predictor + vocoder: units to 16 kHz mono samples in [-1, 1] (empty on failure). */
    external fun synthesize(assetManager: AssetManager, cacheDir: String, units: IntArray): FloatArray
}
