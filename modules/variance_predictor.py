import torch

from params import DurationPredictorParams

class VariancePredictor(torch.nn.Module):
    def __init__(
        self,
        params: DurationPredictorParams,
    ):
        super().__init__()
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                params.embedDim,
                params.hiddenDim,
                kernel_size = params.kernelSize,
                padding = (params.kernelSize - 1) // 2,
            ),
            torch.nn.ReLU(),
        )
        self.ln1 = torch.nn.LayerNorm(params.hiddenDim)
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                params.hiddenDim,
                params.hiddenDim,
                kernel_size = params.kernelSize,
                padding = 1,
            ),
            torch.nn.ReLU(),
        )
        self.ln2 = torch.nn.LayerNorm(params.hiddenDim)
        self.proj = torch.nn.Linear(params.hiddenDim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: B x T x C, Output: B x T
        x = self.conv1(x.transpose(1, 2))
        x = self.conv2(x).transpose(1, 2)
        return self.proj(x).squeeze(dim = 2)