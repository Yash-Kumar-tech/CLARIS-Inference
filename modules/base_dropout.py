# Module kept for compatibility but dropout not required for inference

import torch
from typing import Optional, List, Any

class BaseDropout(torch.nn.Module):
    def __init__(self, p, moduleName = None):
        super().__init__()
        self.p = p
        self.moduleName = moduleName
        self.applyDuringInference = False
        
    def forward(
        self,
        x: torch.Tensor,
        inplace: bool = False
    ):
        return x
    
    def makeGenerationFast_(
        self,
        name: str,
        retainDropout: bool = False,
        retainDropoutModules: Optional[List[str]] = None,
        **kwargs: Any
    ):
        if retainDropout:
            if retainDropoutModules is not None and self.moduleName is None:
                print(
                    f"Cannot enable dropout during inference for module {name} "
                    "because moduleName was not set"
                )
                
        elif (
            retainDropoutModules is None or self.moduleName in retainDropoutModules
        ):
            print(f"Enabling dropout during inference for module {name}")