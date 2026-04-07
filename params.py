import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass 
class DecoderParams:
    attentionHeads: int = 8
    embedDim: int = 256
    ffnEmbedDim: int = 2048
    layerdrop: float = 0.0
    layers: int = 6
    learnedPos: bool = False
    normalizeBefore: bool = True
    outputDim: int = 256
    
@dataclass
class EncoderParams:
    attentionHeads: int = 4
    embedDim: int = 256
    layers: int = 12
    normalizeBefore: bool = True
    ffnEmbedDim: int = 2048

@dataclass
class GenerationParams:
    beam: int = 5
    nbest: int = 1
    noBeamableMm: bool = False
    printAlignment: bool = False
    retainDropout: bool = False
    retainDropoutModules: Optional[List[int]] = None
    
@dataclass
class QuantNoiseParams:
    pq: float = 0.0
    pqBlockSize: int = 0
    
@dataclass
class ModelParams:
    activationFn: str = 'relu'
    activationDropout: float = 0.1
    attentionDropout: float = 0.1
    convChannels: int = 1024
    convKernelSizes: str = "5,5"
    convVersion: str = "s2tTransformer"
    crossSelfAttention: bool = False
    decoder: DecoderParams = field(default_factory=DecoderParams)
    dropout: float = 0.1
    encoder: EncoderParams = field(default_factory=EncoderParams)
    export: bool = False
    generation: GenerationParams = field(default_factory=GenerationParams)
    inputChannels: int = 1
    inputFeatPerChannel: int = 80
    maxSourcePositions: int = 6_000
    maxTargetPositions: int = 5_000
    nFramesPerStep: int = 1
    noScaleEmbedding: bool = False
    noTokenPositionalEmbeddings: bool = False
    quantNoise: QuantNoiseParams = field(default_factory=QuantNoiseParams)
    reluDropout: float = 0.0
    shareDecoderInputOutputEmbed: bool = True
    

class VocoderArgs(argparse.Namespace):
    cpu: bool
    durPrediction: bool
    inCodeFile: str
    resultsPath: str
    vocoder: str
    
@dataclass
class DurationPredictorParams:
    embedDim: int = 128
    hiddenDim: int = 128
    kernelSize: int = 3

@dataclass
class CodeGeneratorParams:
    durPredictorParams: DurationPredictorParams = field(default_factory=DurationPredictorParams)
    embeddingDim: int = 128
    embedderParams: Any = None
    f0: bool = False
    f0QuantNumBin: int = 0
    modelInDim: int = 128
    numEmbeddings: int = 1000
    resblockDilationSizes: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    resblockKernelSizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    upsampleInitialChannels: int = 512
    upsampleRates: List[int] = field(default_factory=lambda: [5, 4, 4, 2, 2])
    upsampleKernelSizes: List[int] = field(default_factory=lambda: [11, 8, 8, 4, 4])

    