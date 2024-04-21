from milligrad.tensor import Tensor

import torch
import numpy as np


def create_randn_pair(shape):
    x_np = np.random.randn(*shape)
    
    x_milli = Tensor(x_np)
    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    
    return x_milli, x_torch

def create_xavier_pair(shape):
    x_np = np.random.randn(*shape) * np.sqrt(2 / np.prod(shape))
    
    x_milli = Tensor(x_np)
    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    
    return x_milli, x_torch