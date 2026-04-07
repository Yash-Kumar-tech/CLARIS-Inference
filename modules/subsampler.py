import torch
from typing import List

class Conv1dSubsampler(torch.nn.Module):
    def __init__(
        self,
        inChannels: int,
        midChannels: int,
        outChannels: int,
        kernelSizes: List[int] = [3, 3]
    ):
        super().__init__()
        self.nLayers = len(kernelSizes)
        self.convLayers = torch.nn.ModuleList(
            torch.nn.Conv1d(
                inChannels if i == 0 else midChannels // 2,
                midChannels if i < self.nLayers - 1 else outChannels * 2,
                k,
                stride = 2,
                padding = k // 2,
            ) for i, k in enumerate(kernelSizes)
        )
        
    def getOutSeqLensTensor(
        self,
        inSeqLensTensor: torch.LongTensor
    ):
        out = inSeqLensTensor.clone()
        for _ in range(self.nLayers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
            
        return out
    
    def forward(
        self,
        srcTokens: torch.Tensor,
        srcLengths: torch.Tensor
    ):
        x = srcTokens.transpose(1, 2).contiguous() # B x T x (C x D) -> # B x (C x D) x T
        for conv in self.convLayers:
            x = conv(x)
            x = torch.nn.functional.glu(x, dim = 1)
        
        x = x.transpose(1, 2).transpose(0, 1).contiguous() # -> T x B x (C x D)
        return x, self.getOutSeqLensTensor(srcLengths)