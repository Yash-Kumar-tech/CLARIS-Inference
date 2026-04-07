import torch
from typing import Iterable

LRELU_SLOPE = 0.1

def getPadding(kernelSize: int, dilation: int) -> int:
    return (kernelSize * dilation - dilation) // 2

class ResBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernelSize: int = 3,
        dilation: Iterable[int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = torch.nn.ModuleList([
            torch.nn.utils.weight_norm(
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernelSize,
                    1,
                    dilation = dilation[i],
                    padding = getPadding(kernelSize, dilation[i])
                )
            ) for i in range(3)
        ])
        
        self.convs2 = torch.nn.ModuleList([
            torch.nn.utils.weight_norm(
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernelSize,
                    1,
                    dilation = 1,
                    padding = getPadding(kernelSize, 1)
                )
            ) for _ in range(3) # Using * 3 would share weights of the conv layers
        ])
        
    def forward(self, x: torch.Tensor):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x
    
    def removeWeightNorm(self):
        for layer in self.convs1:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            torch.nn.utils.remove_weight_norm(layer)