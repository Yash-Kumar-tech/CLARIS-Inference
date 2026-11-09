#ifndef AUDIO_UTILS_H
#define AUDIO_UTILS_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

void ApplyCMVN(std::vector<float> &features, int nMel);

// 80-bin Kaldi fbank (torchaudio.compliance.kaldi.fbank defaults) + per-utterance CMVN, as [frames x 80].
// `waveform` is in int16 scale (float samples multiplied by 32768).
std::vector<float> ComputeFbankCmvn(const float *waveform, size_t size, int sampleRate);

#endif // AUDIO_UTILS_H
