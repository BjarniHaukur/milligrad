from milligrad.tensor import Tensor
from tests.utils import create_randn_pair, create_xavier_pair

import pytest
import torch
import numpy as np


@pytest.mark.parametrize("w1_shape, w2_shape", [
    ((100, 50), (50, 10)),
    ((10, 5), (5, 1)),
    ((768, 512), (512, 2)),
])
def test_perceptron(w1_shape, w2_shape):
    x_milli, x_torch = create_randn_pair((16, w1_shape[0]))
    y_milli, y_torch = create_randn_pair((16, w2_shape[1]))
    
    w1_milli, w1_torch = create_xavier_pair(w1_shape)
    b1_milli, b1_torch = create_randn_pair((w1_shape[1],))
    
    w2_milli, w2_torch = create_xavier_pair(w2_shape)
    b2_milli, b2_torch = create_randn_pair((w2_shape[1],))
    
    # Forward pass
    z1_milli = x_milli @ w1_milli + b1_milli
    z1_torch = x_torch @ w1_torch + b1_torch
    
    a1_milli = z1_milli.relu()
    a1_torch = torch.relu(z1_torch)
    
    z2_milli = a1_milli @ w2_milli + b2_milli
    z2_torch = a1_torch @ w2_torch + b2_torch
    
    loss_milli = ((z2_milli - y_milli)**2).mean()
    loss_torch = ((z2_torch - y_torch)**2).mean()
    
    np.testing.assert_allclose(
        loss_milli.data, loss_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    # Backward pass
    loss_milli.backward()
    loss_torch.backward()
    
    np.testing.assert_allclose(
        x_milli.grad, x_torch.grad.numpy(),
        err_msg="Backward pass input data gradient mismatch", atol=1e-6
    )
    
    np.testing.assert_allclose(
        w1_milli.grad, w1_torch.grad.numpy(),
        err_msg="Backward pass w1 gradient mismatch", atol=1e-6
    )
    
    np.testing.assert_allclose(
        b1_milli.grad, b1_torch.grad.numpy(),
        err_msg="Backward pass b1 gradient mismatch", atol=1e-6
    )
    
    np.testing.assert_allclose(
        w2_milli.grad, w2_torch.grad.numpy(),
        err_msg="Backward pass w2 gradient mismatch", atol=1e-6
    )
    
    np.testing.assert_allclose(
        b2_milli.grad, b2_torch.grad.numpy(),
        err_msg="Backward pass b2 gradient mismatch", atol=1e-6
    )


    
    
