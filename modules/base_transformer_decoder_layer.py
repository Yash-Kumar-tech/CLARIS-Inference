import torch
from typing import Optional, Dict, List

from modules.layer_norm import LayerNorm
from modules.multi_head_attention import MultiheadAttention
from modules.quant_noise import quantNoise
from params import ModelParams
from utils import getActivationFn

class BaseTransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        params: ModelParams,
        noEncoderAttn: bool = False,
        addBiasKv: bool = False,
        addZeroAttn: bool = False
    ):
        super().__init__()
        self.embedDim = params.decoder.embedDim
        self.quantNoise = 0
        self.quantNoiseBlockSize = params.quantNoise.pqBlockSize
        
        self.crossSelfAttention = params.crossSelfAttention
        
        self.selfAttn = self.buildSelfAttention(
            self.embedDim,
            params,
            addBiasKv = addBiasKv,
            addZeroAttn = addZeroAttn
        )
        
        self.nh = self.selfAttn.numHeads
        self.headDim = self.selfAttn.headDim
        
        self.activationFn = getActivationFn(params.activationFn)
        activationDropoutP = params.activationDropout
        
        if activationDropoutP == 0:
            activationDropoutP = params.reluDropout or 0
        self.normalizeBefore = params.decoder.normalizeBefore
        
        self.selfAttnLayerNorm = LayerNorm(
            self.embedDim,
            export = params.export
        )
        
        if noEncoderAttn:
            self.encoderAttn = None
            self.encoderAttnLayerNorm = None
        else:
            self.encoderAttn = self.buildEncoderAttention(
                self.embedDim,
                params
            )
            self.encoderAttnLayerNorm = LayerNorm(
                self.embedDim,
                export = params.export
            )
            
        self.fc1 = self.buildFc(
            self.embedDim,
            params.decoder.ffnEmbedDim,
            self.quantNoise,
            self.quantNoiseBlockSize,
        )
        
        self.fc2 = self.buildFc(
            params.decoder.ffnEmbedDim,
            self.embedDim,
            self.quantNoise,
            self.quantNoiseBlockSize
        )
        
        self.finalLayerNorm = LayerNorm(
            self.embedDim,
            export = params.export
        )
        
        self.needAttn = True
        self.onnxTrace = False
        
    def buildFc(
        self,
        inputDim: int,
        outputDim: int,
        qNoise: float,
        qnBlockSize: int
    ):
        return quantNoise(
            torch.nn.Linear(inputDim, outputDim),
            qNoise,
            qnBlockSize
        )
    
    def buildSelfAttention(
        self,
        embedDim: int,
        params: ModelParams,
        addBiasKv: bool = False,
        addZeroAttn: bool = False,
    ):
        return MultiheadAttention(
            embedDim,
            params.decoder.attentionHeads,
            dropout = params.attentionDropout,
            addBiasKv = addBiasKv,
            addZeroAttn = addZeroAttn,
            selfAttention = not params.crossSelfAttention,
            qNoise = self.quantNoise,
            qnBlockSize = self.quantNoiseBlockSize,
            xformersAttnConfig = None # xformers can be used here
        )
        
    def buildEncoderAttention(self, embedDim, params: ModelParams):
        return MultiheadAttention(
            embedDim,
            params.decoder.attentionHeads,
            kdim = params.encoder.embedDim,
            vdim = params.encoder.embedDim,
            dropout = params.attentionDropout,
            encoderDecoderAttention = True,
            qNoise = self.quantNoise,
            qnBlockSize = self.quantNoiseBlockSize,
            xformersAttnConfig = None # xformers can be used here
        )
        
    def prepareForOnnxExport_(self):
        self.onnxTrace = True
        
    def residualConnection(self, x: torch.Tensor, residual: torch.Tensor):
        return residual + x
    
    def forward(
        self,
        x: torch.Tensor,
        encoderOut: Optional[torch.Tensor] = None,
        encoderPaddingMask: Optional[torch.Tensor] = None,
        incrementalState: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        prevSelfAttnState: Optional[List[torch.Tensor]] = None,
        prevAttnState: Optional[List[torch.Tensor]] = None,
        selfAttnMask: Optional[torch.Tensor] = None,
        selfAttnPaddingMask: Optional[torch.Tensor] = None,
        needAttn: bool = False,
        needHeadWeights: bool = False
    ):
        if needHeadWeights:
            needAttn = True
            
        residual = x
        
        if self.normalizeBefore:
            x = self.selfAttnLayerNorm(x)
        
        if prevSelfAttnState is not None:
            prevKey, prevValue = prevSelfAttnState[:2]
            savedState: Dict[str, Optional[torch.Tensor]] = {
                "prevKey": prevKey,
                "prevValue": prevValue,
            }
            
            if len(prevSelfAttnState) >= 3:
                savedState["prevKeyPaddingMask"] = prevSelfAttnState[2]
            assert incrementalState is not None
            self.selfAttn._setInputBuffer(incrementalState, savedState)
        _selfAttnInputBuffer = self.selfAttn._getInputBuffer(incrementalState)
        if self.crossSelfAttention and not (
            incrementalState is not None
            and _selfAttnInputBuffer is not None
            and "prevKey" in _selfAttnInputBuffer
        ):
            if selfAttnMask is not None:
                assert encoderOut is not None
                selfAttnMask = torch.cat(
                    (x.new_zeros(x.size(0), encoderOut.size(0)), selfAttnMask), dim = 1
                )
                
            if selfAttnPaddingMask is not None:
                if encoderPaddingMask is None:
                    assert encoderOut is not None
                    encoderPaddingMask = selfAttnPaddingMask.new_zeros(
                        encoderOut.size(1), encoderOut.size(0)
                    )
                selfAttnPaddingMask = torch.cat(
                    (encoderPaddingMask, selfAttnPaddingMask), dim = 1
                )
            
            assert encoderOut is not None
            y = torch.cat((encoderOut, x), dim = 0)
        else:
            y = x
         
        x, attn = self.selfAttn(
            query = x,
            key = y,
            value = y,
            keyPaddingMask = selfAttnPaddingMask,
            incrementalState = incrementalState,
            needWeights = False,
            attnMask = selfAttnMask,
        )
        
        x = self.residualConnection(x, residual)
        
        if self.encoderAttn is not None and encoderOut is not None:
            residual = x
            if self.normalizeBefore:
                x = self.encoderAttnLayerNorm(x)
            if prevAttnState is not None:
                prevKey, prevValue = prevAttnState[:2]
                savedState: Dict[str, Optional[torch.Tensor]] = {
                    "prevKey": prevKey,
                    "prevValue": prevValue
                }
                
                if len(prevAttnState) >= 3:
                    savedState["prevKeyPaddingMask"] = prevAttnState[2]
                
                assert incrementalState is not None
                self.encoderAttn._setInputBuffer(incrementalState, savedState)
                
            x, attn = self.encoderAttn(
                query = x,
                key = encoderOut,
                value = encoderOut,
                keyPaddingMask = encoderPaddingMask,
                incrementalState = incrementalState,
                staticKv = True,
                needWeights = needAttn or (not self.training and self.needAttn),
                needHeadWeights = needHeadWeights
            )
            
            x = self.residualConnection(x, residual)
            
            if not self.normalizeBefore:
                x = self.encoderAttnLayerNorm(x)
                
        residual = x
        if self.normalizeBefore:
            x = self.finalLayerNorm(x)
            
        x = self.activationFn(self.fc1(x))
        
        x = self.fc2(x)
        
        
        x = self.residualConnection(x, residual)
        
        
        if not self.normalizeBefore:
            x = self.finalLayerNorm(x)
            
        return x, attn, None