import torch
from modules.res_block import ResBlock
from params import CodeGeneratorParams

LRELU_SLOPE = 0.1

class Generator(torch.nn.Module):
    def __init__(
        self,
        params: CodeGeneratorParams,
    ):
        super().__init__()
        self.numKernels = len(params.resblockKernelSizes)
        self.numUpsamples = len(params.upsampleRates)
        self.convPre = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                params.modelInDim,
                params.upsampleInitialChannels,
                kernel_size = 7,
                stride = 1,
                padding = 3,
            )
        )
        
        self.ups = torch.nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(
                params.upsampleRates,
                params.upsampleKernelSizes
            )
        ):
            self.ups.append(
                torch.nn.utils.weight_norm(
                    torch.nn.ConvTranspose1d(
                        params.upsampleInitialChannels // (2 ** i),
                        params.upsampleInitialChannels // (2 ** (i + 1)),
                        kernel_size = k,
                        stride = u,
                        padding = (k - u) // 2
                    )
                )
            )
        
        self.resblocks = torch.nn.ModuleList()
        
        for i in range(len(self.ups)):
            ch = params.upsampleInitialChannels // (2 ** (i + 1))
            
            for k, d in zip(
                params.resblockKernelSizes,
                params.resblockDilationSizes
            ):
                self.resblocks.append(
                    ResBlock(ch, k, d)
                )
                
        self.convPost = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(ch, 1, 7, 1, padding = 3)
        )
        
    def forward(self, x):
        x = self.convPre(x)
        for i in range(self.numUpsamples):
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.numKernels):
                if xs is None:
                    xs = self.resblocks[i * self.numKernels + j](x)
                else:
                    xs += self.resblocks[i * self.numKernels + j](x)
            x = xs / self.numKernels
        x = torch.nn.functional.leaky_relu(x)
        x = self.convPost(x)
        x = torch.tanh(x)
        
        return x
    
    def removeWeightNorm(self):
        for layer in self.ups:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            assert isinstance(layer, ResBlock)
            layer.removeWeightNorm()
        
        torch.nn.utils.remove_weight_norm(self.convPre)
        torch.nn.utils.remove_weight_norm(self.convPost)