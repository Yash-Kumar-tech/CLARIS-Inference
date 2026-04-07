import torch.nn as nn

def Linear(inFeatures: int, outFeatures: int, bias: bool = True):
    m = nn.Linear(inFeatures, outFeatures, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m