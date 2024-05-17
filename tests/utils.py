from milligrad.tensor import Tensor

import torch
import torch.nn.functional as F
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

def create_one_hot_pair(shape, num_classes):
    labels_np = np.random.randint(0, num_classes, shape)
    one_hot_np = np.eye(num_classes)[labels_np]
    
    one_hot_milli = Tensor(one_hot_np)
    one_hot_torch = torch.tensor(one_hot_np, dtype=torch.float32, requires_grad=True)
    
    return one_hot_milli, one_hot_torch