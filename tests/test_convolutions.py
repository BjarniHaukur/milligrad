from milligrad.tensor import Tensor
from tests.utils import create_randn_pair

import pytest

import torch
import torch.nn.functional as F
import numpy as np

@pytest.mark.parametrize("batch_size, seq_len, c_in, c_out, kernel_size, padding, stride", [
    (1, 1, 1, 1, 1, 0, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (16, 100, 3, 4, 3, 2, 3),
    (16, 3, 1, 4, 3, 0, 3),
    (16, 5, 3, 9, 5, 0, 2),
    (16, 14, 3, 10, 7, 100, 2),
])  
def test_conv1d(batch_size, seq_len, c_in, c_out, kernel_size, padding, stride):
    kernels_milli, kernels_torch = create_randn_pair((c_in, kernel_size, c_out))
    kernels_torch = kernels_torch.permute(2,0,1) # PyTorch uses (out_channels, in_channels, wH)
    x_milli, x_torch = create_randn_pair((batch_size, c_in, seq_len))

    conv_milli = x_milli.conv1d(kernels_milli, padding=padding, stride=stride)
    conv_torch = F.conv1d(x_torch, kernels_torch, padding=padding, stride=stride)
    
    np.testing.assert_allclose(
        conv_milli.data, conv_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-5
    )
    
    conv_milli.backward()
    conv_torch.backward(torch.ones_like(conv_torch))
    
    np.testing.assert_allclose(
        x_milli.grad, x_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-5
    )    
    
@pytest.mark.parametrize("batch_size, height, width, c_in, c_out, kernel_size, padding", [
    (1, 1, 1, 1, 1, (1, 1), (0, 0)),
    (32, 28, 28, 1, 3, (3, 3), (0, 0)),
    (32, 28, 28, 1, 3, (3, 3), (1, 1)),
    (32, 28, 28, 3, 3, (5, 5), (1, 1)),
    (32, 32, 32, 3, 16, (5, 5), (2, 2)),
])
def test_conv2d(batch_size, height, width, c_in, c_out, kernel_size, padding):
    kernels_milli, kernels_torch = create_randn_pair((c_in, *kernel_size, c_out))
    kernels_torch = kernels_torch.permute(3,0,1,2) # PyTorch uses (out_channels, in_channels, kH, wH)
    x_milli, x_torch = create_randn_pair((batch_size, c_in, height, width))
    
    conv_milli = x_milli.conv2d(kernels_milli, padding=padding)
    conv_torch = F.conv2d(x_torch, kernels_torch, padding=padding)
    
    np.testing.assert_allclose(
        conv_milli.data, conv_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=2e-5
    )
    
    conv_milli.backward()
    conv_torch.backward(torch.ones_like(conv_torch))
    
    np.testing.assert_allclose(
        x_milli.grad, x_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=2e-5
    )

@pytest.mark.parametrize("batch_size, channels, seq_len, kernel_size", [
    (1, 1, 2, 2),
    (1, 1, 10, 5),
    (16, 3, 100, 2),
    (16, 3, 99, 3),
    (16, 3, 70, 7),
    (16, 3, 50, 10),
])
def test_maxpool1d(batch_size, channels, seq_len, kernel_size):
    x_milli, x_torch = create_randn_pair((batch_size, channels, seq_len))

    pooled_milli = x_milli.maxpool1d(kernel_size=kernel_size)
    pooled_torch = F.max_pool1d(x_torch, kernel_size=kernel_size)

    np.testing.assert_allclose(
        pooled_milli.data, pooled_torch.detach().numpy(),
        err_msg="Forward pass mismatch in maxpool1d", atol=1e-5
    )

    pooled_milli.backward()
    pooled_torch.backward(torch.ones_like(pooled_torch))

    np.testing.assert_allclose(
        x_milli.grad, x_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch in maxpool1d", atol=1e-5
    )

@pytest.mark.parametrize("batch_size, channels, height, width, kernel_size", [
    (1, 1, 2, 2, (2, 2)),
    (1, 1, 8, 8, (2, 2)),
    (16, 3, 28, 28, (4, 4)),
    (16, 3, 28, 28, (7, 7)),
    (16, 3, 32, 32, (8, 8)),
    (32, 3, 32, 32, (16, 16)),
])
def test_maxpool2d(batch_size, channels, height, width, kernel_size):
    x_milli, x_torch = create_randn_pair((batch_size, channels, height, width))

    pooled_milli = x_milli.maxpool2d(kernel_size=kernel_size)
    pooled_torch = F.max_pool2d(x_torch, kernel_size=kernel_size)

    np.testing.assert_allclose(
        pooled_milli.data, pooled_torch.detach().numpy(),
        err_msg="Forward pass mismatch in maxpool2d", atol=1e-5
    )

    pooled_milli.backward()
    pooled_torch.backward(torch.ones_like(pooled_torch))

    np.testing.assert_allclose(
        x_milli.grad, x_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch in maxpool2d", atol=1e-5
    )
    
if __name__ == "__main__":
    test_conv1d(16, 100, 3, 4, 3, 2, 3)
    test_maxpool2d(1, 1, 4, 4, (2, 2))
    test_maxpool2d(32, 3, 32, 32, (16, 16))