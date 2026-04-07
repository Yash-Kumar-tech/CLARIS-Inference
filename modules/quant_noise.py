# This module simulates quantization noise during training
# Ideally, improves performance for quantized models 
# However, we've been running the model at FP32 with decent speeds
# Therefore, this module functions as a dummy wrapper
# and complete implementation is left for future work

import torch

def quantNoise(module, p, blockSize):
    return module