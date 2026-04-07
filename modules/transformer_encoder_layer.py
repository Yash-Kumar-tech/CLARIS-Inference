import torch
from typing import List, Optional

from modules.base_dropout import BaseDropout
from modules.layer_norm import LayerNorm
from modules.multi_head_attention import MultiheadAttention
from modules.quant_noise import quantNoise
from params import ModelParams
from utils import getActivationFn

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        params: ModelParams,
        returnFc: bool = False
    ):
        super().__init__()
        self.params = params
        self.returnFc = returnFc
        self.embedDim = params.encoder.embedDim
        self.quantNoise = params.quantNoise.pq
        self.quantNoiseBlockSize = params.quantNoise.pqBlockSize
        self.selfAttn = self.buildSelfAttention(self.embedDim, params)
        self.selfAttnLayerNorm = LayerNorm(
            self.embedDim,
            export = params.export
        )
        self.activationFn = getActivationFn(activation = params.activationFn)
        
        self.normalizeBefore = params.encoder.normalizeBefore
        self.fc1 = self.buildFc(
            self.embedDim,
            params.encoder.ffnEmbedDim,
            self.quantNoise,
            self.quantNoiseBlockSize
        )
        
        self.fc2 = self.buildFc(
            params.encoder.ffnEmbedDim,
            self.embedDim,
            self.quantNoise,
            self.quantNoiseBlockSize
        )
        
        self.finalLayerNorm = LayerNorm(self.embedDim, export = params.export)

    def buildFc(
        self,
        inputDim: int,
        outputDim: int,
        qNoise: float,
        qnBlockSize: int
    ) -> torch.nn.Linear:
        return quantNoise(
            torch.nn.Linear(
                inputDim,
                outputDim
            ),
            p = qNoise,
            blockSize = qnBlockSize
        )
        
    def _getFcRank(self, removeNum: int) -> List[int]:
        f1FilterParam = []
        for i in range(self.fc1.out_features):
            f1FilterParam.append(
                torch.sum(torch.abs(self.fc1.weight[i]))
                + torch.sum(torch.abs(self.fc2.weight[:, i]))
                + torch.abs(self.fc1.bias[i])
            )
        
        return sorted(
            range(len(f1FilterParam)), 
            key = lambda k: f1FilterParam[k], 
            reverse = False
        )[0: removeNum]
        
    def buildSelfAttention(self, embedDim: int, params: ModelParams):
        return MultiheadAttention(
            embedDim,
            params.encoder.attentionHeads,
            dropout = 0,
            selfAttention = True,
            qNoise = self.quantNoise,
            qnBlockSize = self.quantNoiseBlockSize,
            xformersAttnConfig = None # xformers can be used here
        )
        
    def residualConnection(self, x: torch.Tensor, residual: torch.Tensor):
        return residual + x
    
    def forward(
        self,
        x: torch.Tensor,
        encoderPaddingMask: Optional[torch.Tensor],
        attnMask: Optional[torch.Tensor] = None
    ):
        if attnMask is not None:
            attnMask = attnMask.masked_fill(
                attnMask.to(torch.bool),
                -1e8 if x.dtype == torch.float32 else -1e4
            )
            
        residual = x
        if self.normalizeBefore:
            x = self.selfAttnLayerNorm(x)
        
        x, _ = self.selfAttn(
            query = x,
            key = x,
            value = x,
            keyPaddingMask = encoderPaddingMask,
            needWeights = False,
            attnMask = attnMask
        )
        
        x = self.residualConnection(x, residual)
        
        if not self.normalizeBefore:
            x = self.selfAttnLayerNorm(x)
            
        residual = x
        if self.normalizeBefore:
            x = self.finalLayerNorm(x)
            
        x = self.activationFn(self.fc1(x))
        x = self.fc2(x)
        
        fcResult = x
        
        x = self.residualConnection(x, residual)
        if not self.normalizeBefore:
            x = self.finalLayerNorm(x)
            
        if self.returnFc and not torch.jit.is_scripting():
            return x, fcResult
        return x