# Module kept for compatibility but dropout not required for inference

import torch

class LayerDropModuleList(torch.nn.ModuleList):
    def __init__(self, p, modules = None):
        super().__init__(modules)
        self.p = p
    
    def __iter__(self):
        dropoutProbs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropoutProbs[i] > self.p):
                yield m