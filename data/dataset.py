from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import os
import soundfile as sf
import torchaudio.sox_effects as taSox
import torchaudio.compliance.kaldi as taKaldi

from data.transforms import UtteranceCMVN

@dataclass
class DataItem:
    index: int
    source: torch.Tensor
    filename: str
    
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        nFramesPerStep: int
    ):
        super().__init__()
        self.root = root
        self.files = [f for f in os.listdir(self.root) if f.endswith('.wav')]
        self.nFramesPerStep = nFramesPerStep
        self.featureTransforms = UtteranceCMVN()
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> DataItem:
        filename = self.files[index]
        source = self.getSourceAudio(filename)
        return DataItem(
            index = index,
            filename = filename,
            source = source
        )
        
    def getSourceAudio(self, filename: str) -> torch.Tensor:
        source = getFeaturesOrWaveform(
            os.path.join(self.root, filename),
            needWaveform = False,
            useSampleRate = 16_000,
        )
        
        source = self.featureTransforms(source)
        source = torch.from_numpy(source).float()
        return source
    
    def collater(
        self,
        samples: List[DataItem],
        returnOrder: bool = False
    ) -> Dict:
        indices = torch.tensor([x.index for x in samples], dtype = torch.long)
        frames = _collateFrames([x.source for x in samples])
        filenames = [f.filename for f in samples]
        nFrames = torch.tensor([x.source.size(0) for x in samples], dtype = torch.long)
        nFrames, order = nFrames.sort(descending = True)
        
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)
        
        netInput = {
            "srcTokens": frames,
            "srcLengths": nFrames
        }
        
        out = {
            "id": indices,
            "filenames": filenames,
            "netInput": netInput,
        }
        
        if returnOrder:
            out["order"] = order
        
        return out

def _getFBank(
    waveform: np.ndarray,
    sampleRate: int,
    nBins: int
):
    waveform = torch.from_numpy(waveform)
    features = taKaldi.fbank(
        waveform,
        num_mel_bins = nBins,
        sample_frequency = sampleRate
    )
    
    return features.numpy()
    
def getFeaturesOrWaveform(
    path: str,
    needWaveform: bool = False,
    useSampleRate: int = 16_000
) -> np.ndarray:
    waveform, sampleRate = getWaveform(
        path,
        normalization = False
    )
    
    features = _getFBank(
        waveform = waveform,
        sampleRate = sampleRate,
        nBins = 80,
    )
    
    return features

def getWaveform(
    path: str,
    normalization: bool = True,
    mono: bool = True,
    frames: int = -1,
    start: int = 0,
    always2d: bool = True,
    outputSampleRate: Optional[int] = None,
    normalizeVolume: bool = False,
    waveformTransforms: Any = None
) -> Tuple[np.ndarray, int]:
    waveform, sampleRate = sf.read(
        path,
        dtype = "float32",
        always_2d = always2d,
        frames = frames,
        start = start,
    )
    
    waveform = waveform.T
    waveform, sampleRate = convertWaveform(
        waveform,
        sampleRate,
        normalizeVolume = normalizeVolume,
        toMono = mono,
        toSampleRate = outputSampleRate,
    )
    
    if not normalization:
        waveform *= (2 ** 15)
        
    if not always2d:
        waveform = waveform.squeeze(axis = 0)
        
    return waveform, sampleRate

def convertWaveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sampleRate: int,
    normalizeVolume: bool = False,
    toMono: bool = False,
    toSampleRate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    effects = []
    
    if normalizeVolume:
        effects.append({'gain', '-n'})
    if toSampleRate is not None and toSampleRate != sampleRate:
        effects.append(["rate", f"{toSampleRate}"])
    if toMono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
        
    if len(effects) > 0:
        isNpInput = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if isNpInput else waveform
        converted, convertedSampleRate = taSox.apply_effects_tensor(
            _waveform,
            sampleRate,
            effects,
        )
        
        if isNpInput:
            converted = converted.numpy()
        
        return converted, convertedSampleRate
    return waveform, sampleRate

def _collateFrames(
    frames: List[torch.Tensor],
    isAudioInput: bool = False
) -> torch.Tensor:
    maxLen = max(frame.size(0) for frame in frames)
    if isAudioInput:
        raise NotImplementedError("Not supporting audio input as of yet")
    else:
        out = frames[0].new_zeros((len(frames), maxLen, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
        
    return out