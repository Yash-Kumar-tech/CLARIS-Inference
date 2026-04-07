# Custom wrapper over torch.nn.LayerNorm, will help with torch.export later on

import torch

def LayerNorm(normalizedShape, eps = 1e-5, elementwiseAffine = True, export = False):
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        export = True
    return torch.nn.LayerNorm(normalizedShape, eps, elementwiseAffine)