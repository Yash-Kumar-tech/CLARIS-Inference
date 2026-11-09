#include "audio_utils.h"

#include "kaldi-native-fbank/csrc/online-feature.h"

void ApplyCMVN(std::vector<float>& features, int nMel) {
    int nFrames = features.size() / nMel;
    if (nFrames <= 1) return;

    std::vector<float> means(nMel, 0.0f);
    std::vector<float> vars(nMel, 0.0f);

    for (int f = 0; f < nFrames; ++f) {
        for (int m = 0; m < nMel; ++m) {
            means[m] += features[f * nMel + m];
        }
    }
    for (int m = 0; m < nMel; ++m) means[m] /= nFrames;

    for (int f = 0; f < nFrames; ++f) {
        for (int m = 0; m < nMel; ++m) {
            float diff = features[f * nMel + m] - means[m];
            vars[m] += diff * diff;
        }
    }
    for (int m = 0; m < nMel; ++m) {
        vars[m] = std::sqrt(vars[m] / nFrames);
        if (vars[m] < 1e-5f) vars[m] = 1e-5f;
    }

    for (int f = 0; f < nFrames; ++f) {
        for (int m = 0; m < nMel; ++m) {
            features[f * nMel + m] = (features[f * nMel + m] - means[m]) / vars[m];
        }
    }
}

std::vector<float> ComputeFbankCmvn(const float* waveform, size_t size, int sampleRate) {
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = sampleRate;
    opts.frame_opts.frame_shift_ms = 10.0f;
    opts.frame_opts.frame_length_ms = 25.0f;
    opts.frame_opts.dither = 0.0f; // matches python temp_mel.py torchaudio defaults
    opts.frame_opts.preemph_coeff = 0.97f;
    opts.frame_opts.remove_dc_offset = true;
    opts.frame_opts.window_type = "povey"; // default in python torchaudio compliance
    opts.frame_opts.snip_edges = true;
    opts.mel_opts.num_bins = 80;
    opts.mel_opts.low_freq = 20.0f;
    opts.mel_opts.high_freq = 0.0f;
    opts.mel_opts.vtln_low = 100.0f;
    opts.mel_opts.vtln_high = -500.0f;
    opts.use_energy = false;

    knf::OnlineFbank fbank(opts);
    fbank.AcceptWaveform(sampleRate, waveform, size);
    fbank.InputFinished();

    int numFrames = fbank.NumFramesReady();
    std::vector<float> melSpec(numFrames * 80);
    for (int i = 0; i < numFrames; ++i) {
        const float* frame = fbank.GetFrame(i);
        std::copy(frame, frame + 80, melSpec.begin() + i * 80);
    }

    ApplyCMVN(melSpec, 80);
    return melSpec;
}
