import torch
from typing import Dict, Iterable, List, Optional
import sys

from model import SpeechToUnitTransformer

class EnsembleModel(torch.nn.Module):
    def __init__(self, model: SpeechToUnitTransformer):
        super().__init__()
        self.modelsSize = 1
        self.model = model
        self.hasIncremental: bool = True
        
    def forward(self):
        pass
    
    def hasEncoder(self):
        return True
    
    def hasIncrementalStates(self):
        return self.hasIncremental
    
    def maxDecoderPositions(self):
        return min(self.model.maxDecoderPositions(), sys.maxsize)
    
    def setDecoderBeamSize(self, beamSize):
        if beamSize > 1:
            self.model.setBeamSize(beamSize)
            
    @torch.jit.export
    def forwardEncoder(
        self,
        srcTokens: torch.Tensor,
        srcLengths: torch.tensor
    ):
        return self.model.encoder.forwardTorchscript(
            srcTokens = srcTokens,
            srcLengths = srcLengths
        )
        
    @torch.jit.export
    def forwardDecoder(
        self,
        tokens: torch.Tensor,
        encoderOut: torch.Tensor,
        encoderPaddingMask: torch.Tensor,
        incrementalStates: Dict[str, Dict[str, Optional[torch.Tensor]]],
        temperature: float = 1.0
    ):
        decoderOut = self.model.decoder.forward(
            prevOutputTokens = tokens,
            encoderOut = encoderOut,
            encoderPaddingMask = encoderPaddingMask,
            incrementalState = incrementalStates[0],
        )
        
        attn: Optional[torch.Tensor] = None
        decoderLen = len(decoderOut)
        
        if decoderLen > 1 and decoderOut[1] is not None: # Almost always True
            if isinstance(decoderOut[1], torch.Tensor):
                attn = decoderOut[1][0]
            else:
                attnHolder = decoderOut[1][0]
                attn = attnHolder[0]
            if attn is not None:
                attn = attn[:, -1, :]
        
        decoderOutTuple = (
            decoderOut[0][:, -1:, :].div_(temperature), 
            None if decoderLen <= 1 else decoderOut[1]
        )
        probs = self.model.getNormalizedProbs(decoderOutTuple, logProbs = True, sample = None)
        probs = probs[:, -1, :]
        return probs, attn
    
    @torch.jit.export
    def reorderEncoderOut(
        self,
        encoderOut: Optional[torch.Tensor] = None,
        encoderPaddingMask: Optional[torch.Tensor] = None,
        encoderEmbedding: Optional[torch.Tensor] = None,
        encoderStates: Optional[torch.Tensor] = None,
        newOrder: Optional[torch.Tensor] = None,
    ):
        return self.model.encoder.reorderEncoderOut(
            encoderOut = encoderOut,
            encoderPaddingMask = encoderPaddingMask,
            encoderEmbedding = encoderEmbedding,
            encoderStates = encoderStates,
            newOrder = newOrder
        )
        
    @torch.jit.export
    def reorderIncrementalState(
        self,
        incrementalState: Dict[str, Dict[str, Optional[torch.Tensor]]],
        newOrder,
    ):
        self.model.decoder.reorderIncrementalStateScripting(incrementalState, newOrder)