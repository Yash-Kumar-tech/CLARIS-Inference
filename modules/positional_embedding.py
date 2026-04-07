import torch
from typing import Optional, Any
import math

from utils import makePositions

class LearnedPositionalEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        numEmbeddings: int,
        embeddingDim: int,
        paddingIdx: int
    ):
        super().__init__(numEmbeddings, embeddingDim, paddingIdx)
        
        self.onnxTrace = False
        if self.padding_idx is not None:
            self.maxPositions = self.num_embeddings - self.padding_idx - 1
        else:
            self.maxPositions = self.num_embeddings
    
    def forward(
        self,
        input: torch.Tensor,
        isIncrementalStateNone: bool = True,
        positions: Optional[torch.Tensor] = None
    ):
        assert (positions is None) or self.padding_idx is None, (
            "If positions is pre-computed, then padding_idx should not be set"
        )
        
        if positions is None:
            if isIncrementalStateNone == False:
                positions = torch.zeros(
                    (1, 1),
                    device = input.device,
                    dtype = input.dtype
                ).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = makePositions(
                    input,
                    self.padding_idx,
                    onnx_trace = self.onnxTrace
                )
                
        return torch.nn.functional.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse
        )
    
class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        embedddingDim: int,
        paddingIdx: int,
        initSize: int = 1024,
        autoExpand: bool = True
    ):
        super().__init__()
        self.embeddingDim = embedddingDim
        self.paddingIdx = paddingIdx if paddingIdx is not None else 0
        self.register_buffer(
            "weights",
            SinusoidalPositionalEmbedding.getEmbedding(
                initSize,
                embedddingDim,
                paddingIdx
            ),
            persistent = False,
        )
        
        self.maxPositions = int(1e5)
        self.autoExpand = autoExpand
        self.onnxTrace = False
        
    def prepareForOnnxExport_(self):
        self.onnxTrace = True
    
    @staticmethod
    def getEmbedding(
        numEmbeddings: int,
        embeddingDim: int,
        paddingIdx: Optional[int] = None
    ):
        halfDim = embeddingDim // 2
        emb = math.log(10_000) / (halfDim - 1)
        emb = torch.exp(torch.arange(halfDim, dtype = torch.float) * -emb)
        emb = torch.arange(numEmbeddings, dtype = torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim = 1).view(numEmbeddings, -1)
        
        if embeddingDim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(numEmbeddings, 1)], dim = 1)
        if paddingIdx is not None:
            emb[paddingIdx, :] = 0
        return emb
    
    def forward(
        self,
        input: torch.Tensor,
        isIncrementalStateNone: bool = True,
        timestep: Optional[torch.Tensor] = None,
        positions: Optional[Any] = None
    ):
        bspair = torch._shape_as_tensor(input)
        bsz, seqLen = bspair[0], bspair[1]
        
        maxPos = self.paddingIdx + 1 + seqLen
        weights = self.weights
        
        if maxPos > self.weights.size(0):
            weights = SinusoidalPositionalEmbedding.getEmbedding(
                maxPos,
                self.embeddingDim,
                self.paddingIdx
            ).to(self.weights)
            
            if self.autoExpand:
                self.weights = weights
        if isIncrementalStateNone == False:
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seqLen
            if self.onnxTrace:
                return (
                    weights.index_select(index = self.paddingIdx + pos, dim = 0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return weights[self.paddingIdx + pos, :].expand(bsz, 1, -1)
        
        positions = makePositions(
            input,
            self.paddingIdx,
            onnxTrace = self.onnxTrace
        )
        
        if self.onnxTrace:
            flatEmbeddings = weights.detach().index_select(0, positions.view(-1))
            embeddingShape = torch.cat((
                bsz.view(1), seqLen.view(1), torch.Tensor([-1], dtype = torch.long)
            ))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flatEmbeddings,
                embeddingShape
            )
            
            return embeddings
        return (
            weights.index_select(0, positions.view(-1)).view(bsz, seqLen, -1).detach()
        )    
        
def PositionalEmbedding(
    numEmbeddings: int,
    embeddingDim: int,
    paddingIdx: int,
    learned: bool = False,
    autoExpand: bool = True
):
    if learned:
        if paddingIdx is not None:
            numEmbeddings = numEmbeddings + paddingIdx + 1
        
        m = LearnedPositionalEmbedding(numEmbeddings, embeddingDim, paddingIdx)
        torch.nn.init.normal_(m.weight, mean = 0, std = embeddingDim ** -0.5)
        if paddingIdx is not None:
            torch.nn.init.constant_(m.weight[paddingIdx], 0)
            
    else:
        m = SinusoidalPositionalEmbedding(
            embeddingDim,
            paddingIdx,
            initSize = numEmbeddings + paddingIdx + 1,
            autoExpand = autoExpand,
        )
        
    return m

