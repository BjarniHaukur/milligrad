from milligrad.tensor import Tensor

import pytest

import torch
import torch.nn.functional as F
import numpy as np

@pytest.mark.parametrize("batch_size, seq_len, c_in, c_out, kernel_size, padding", [
    (1, 1, 1, 1, 1, 0),
    (1, 1, 1, 1, 1, 1),
    (16, 100, 3, 4, 3, 2),
    (16, 5, 3, 9, 5, 0),
    (16, 14, 3, 10, 7, 100),
])  
def test_conv1d(batch_size, seq_len, c_in, c_out, kernel_size, padding):
    kernels_np = np.random.randn(c_in, kernel_size, c_out) # (in_channels, kernel_size, out_channels)
    x_np = np.random.randn(batch_size, c_in, seq_len) # (batch_size, in_channels, sequence_length)
    
    kernels = Tensor(kernels_np)
    x = Tensor(x_np)
    
    # torch does (out_channels, in_channels, kernel_size)
    kernels_torch = torch.tensor(kernels_np.transpose(2,0,1), requires_grad=True)
    x_torch = torch.tensor(x_np, requires_grad=True)
    
    conv_torch = F.conv1d(x_torch, kernels_torch, padding=padding)
    conv_milligrad = x.conv1d(kernels, padding=padding)
    
    np.testing.assert_allclose(
        conv_milligrad.data, conv_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    conv_torch.backward(torch.ones_like(conv_torch))
    conv_milligrad.backward()
    
    np.testing.assert_allclose(
        x.grad, x_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )    
    
@pytest.mark.parametrize("batch_size, seq_len, c_in, c_out, kernel_size, padding", [
    (1, 1, 1, 1, (1,1), (0,0)),
    (1, 1, 1, 1, (1,1), (1,1)),
    # (16, 100, 3, 4, (3,3), (2,2)),
])
def test_conv2d(batch_size, seq_len, c_in, c_out, kernel_size, padding):
    kernels_np = np.random.randn(c_in, *kernel_size, c_out)
    x_np = np.random.randn(batch_size, c_in, seq_len, seq_len)
    
    kernels = Tensor(kernels_np)
    x = Tensor(x_np)
    
    kernels_torch = torch.tensor(kernels_np.transpose(3,0,1,2), requires_grad=True)
    x_torch = torch.tensor(x_np, requires_grad=True)
    
    conv_torch = F.conv2d(x_torch, kernels_torch, padding=padding)
    conv_milligrad = x.conv2d(kernels, padding=padding)
    
    np.testing.assert_allclose(
        conv_milligrad.data, conv_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    # TODO: implement backward pass for conv2d and test it