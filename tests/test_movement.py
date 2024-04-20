from milligrad.tensor import Tensor
from tests.utils import create_randn_pair

import pytest

import torch
import numpy as np

@pytest.mark.parametrize("shape", [
    (2,),
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 5),
])
def test_transpose(shape):
    __import__("warnings").filterwarnings("ignore", message="The use of `x.T`")
    a_milli, a_torch = create_randn_pair(shape)
    
    a_milli_t = a_milli.T
    a_torch_t = a_torch.T
    
    np.testing.assert_allclose(
        a_milli_t.data, a_torch_t.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    a_milli_t.backward()
    a_torch_t.backward(torch.ones_like(a_torch_t))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
@pytest.mark.parametrize("old_shape, new_shape", [
    ((2, 3, 4), (3, 2, 4)),
    ((2, 3, 4), (2, 3, 4)),
    ((2, 3, 4), (4, 3, 2)),
    ((2, 3, 4), (2, 1, 3, 4)),
    ((2, 3, 4), (1, 2, 3, 4)),
    ((2, 3, 4), (1, 1, 2, 3, 4)),
])
def test_reshape(old_shape, new_shape):
    a_milli, a_torch = create_randn_pair(old_shape)
    
    a_milli_reshaped = a_milli.reshape(new_shape)
    a_torch_reshaped = a_torch.reshape(new_shape)
    
    np.testing.assert_allclose(
        a_milli_reshaped.data, a_torch_reshaped.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    a_milli_reshaped.backward()
    a_torch_reshaped.backward(torch.ones_like(a_torch_reshaped))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )