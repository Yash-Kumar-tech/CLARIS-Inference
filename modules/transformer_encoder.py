import torch
from typing import Dict, Optional
import math

from modules.subsampler import Conv1dSubsampler
from modules.positional_embedding import PositionalEmbedding
from modules.layer_norm import LayerNorm
from modules.transformer_encoder_layer import TransformerEncoderLayer
from params import ModelParams
from utils import lengthsToPaddingMask

class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        params: ModelParams,
    ):
        super().__init__()
        self.dictionary = None
        self.numUpdates = 0
        self.embedScale = math.sqrt(params.encoder.embedDim)
        
        if params.noScaleEmbedding:
            self.embedScale = 1.0
        self.paddingIdx = 1
        
        self.convVersion = params.convVersion
        
        if self.convVersion == "s2tTransformer":
            self.subsample = Conv1dSubsampler(
                params.inputFeatPerChannel * params.inputChannels,
                params.convChannels,
                params.encoder.embedDim,
                [int(k) for k in params.convKernelSizes.split(",")],
            )
        else:
            raise NotImplementedError(f"Invalid conv version ({self.convVersion})")
        
        self.embedPositions = PositionalEmbedding(
            params.maxSourcePositions,
            params.encoder.embedDim,
            self.paddingIdx
        )
        
        self.transformerLayers = torch.nn.ModuleList([
            TransformerEncoderLayer(params) for _ in range(params.encoder.layers)
        ])
        
        self.layerNorm = LayerNorm(params.encoder.embedDim)
    
    def forwardTorchscript(
        self, 
        srcTokens: torch.LongTensor,
        srcLengths: torch.LongTensor
    ):
        if torch.jit.is_scripting():
            return self.forward(
                srcTokens = srcTokens,
                srcLengths = srcLengths
            )
        return self.forwardNonTorchscript(srcTokens, srcLengths)
    
    @torch.jit.unused
    def forwardNonTorchscript(
        self,
        srcTokens: torch.LongTensor,
        srcLengths: torch.LongTensor
    ):
        return self.forward(srcTokens, srcLengths)
    
    def maxPositions(self):
        return int(1e6)
    
    def setNumUpdates(self, numUpdates: int):
        def _apply(m):
            if hasattr(m, "setNumUpdates") and m != self:
                m.setNumUpdates(numUpdates)
        
        self.apply(_apply)
        self.numUpdates = numUpdates
        
    def _forward(
        self,
        srcTokens: torch.LongTensor,
        srcLengths: torch.LongTensor,
        returnAllHiddens: bool = False
    ):
        x, inputLengths = self.subsample(srcTokens, srcLengths)
        x = self.embedScale * x
        
        encoderPaddingMask = lengthsToPaddingMask(inputLengths)
        positions = self.embedPositions(encoderPaddingMask).transpose(0, 1)
        x += positions
        
        encoderStates = []
        
        for layer in self.transformerLayers:
            x = layer(x, encoderPaddingMask)
            if returnAllHiddens:
                encoderStates.append(x)
            
        if self.layerNorm is not None:
            x = self.layerNorm(x)

        return (
            x.unsqueeze(0),
            encoderPaddingMask.unsqueeze(0) if encoderPaddingMask.any() else torch.empty(0),
            encoderStates,
        )
        
    def forward(
        self,
        srcTokens: torch.LongTensor,
        srcLengths: torch.LongTensor,
        returnAllHiddens: bool = False
    ):
        x = self._forward(srcTokens, srcLengths, returnAllHiddens)
        return x
    
    def reorderEncoderOut(
        self,
        encoderOut: Optional[torch.Tensor] = None, 
        encoderPaddingMask: Optional[torch.Tensor] = None,
        encoderEmbedding: Optional[torch.Tensor] = None,
        encoderStates: Optional[torch.Tensor] = None,
        newOrder: Optional[torch.Tensor] = None,
    ):
        newEncoderOut = (
            None if encoderOut is None or len(encoderOut) == 0 
            else encoderOut.index_select(2, newOrder)
        )
        
        newEncoderPaddingMask = (
            None if encoderPaddingMask is None or len(encoderPaddingMask) == 0
            else encoderPaddingMask.index_select(1, newOrder)
        )
        
        newEncoderEmbedding = (
            None if encoderEmbedding is None or len(encoderEmbedding) == 0
            else encoderEmbedding.index_select(1, newOrder)
        )
        
        newEncoderStates = encoderStates
        if encoderStates is not None and len(encoderStates) > 0:
            for idx in range(encoderStates.size(0)):
                newEncoderStates[idx] = encoderStates[idx].index_select(1, newOrder)
            
        return (
            newEncoderOut,
            newEncoderPaddingMask,
            newEncoderEmbedding,
            newEncoderStates
        )