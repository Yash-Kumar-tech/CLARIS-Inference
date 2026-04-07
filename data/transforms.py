import torch
import numpy as np

class UtteranceCMVN:
    def __init__(
        self,
        normMeans: bool = True,
        normVars: bool = True
    ):
        self.normMeans, self.normVars = normMeans, normVars
        
    def __call__(self, x: np.ndarray):
        mean = x.mean(axis = 0)
        squareSums = (x ** 2).sum(axis = 0)
        
        if self.normMeans:
            x = np.subtract(x, mean)
        if self.normVars:
            var = squareSums / x.shape[0] - mean ** 2
            std = np.sqrt(np.maximum(var, 1e-10))
            x = np.divide(x, std)
        
        return x