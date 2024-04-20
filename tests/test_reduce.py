from milligrad.tensor import Tensor
from tests.utils import create_randn_pair

import pytest

import torch
import torch.nn.functional as F
import numpy as np

shapes_and_axis = [
    ((2,), None),
    ((2,), 0),
    ((2, 3), None),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3, 4), None),
    ((2, 3, 4), 0),
    ((2, 3, 4), 1),
    ((2, 3, 4), 2),
]

@pytest.mark.parametrize("shape, axis", shapes_and_axis)
def test_sum(shape, axis):
    a_milli, a_torch = create_randn_pair(shape)
    
    sum_milli = a_milli.sum(axis=axis)
    sum_torch = a_torch.sum(dim=axis)
    
    np.testing.assert_allclose(
        sum_milli.data, sum_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    sum_milli.backward()
    sum_torch.backward(torch.ones_like(sum_torch))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
@pytest.mark.parametrize("shape, axis", shapes_and_axis)
def test_mean(shape, axis):
    a_milli, a_torch = create_randn_pair(shape)
    
    mean_milli = a_milli.mean(axis=axis)
    mean_torch = a_torch.mean(dim=axis)
    
    np.testing.assert_allclose(
        mean_milli.data, mean_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    mean_milli.backward()
    mean_torch.backward(torch.ones_like(mean_torch))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )

@pytest.mark.parametrize("shape, axis, unbiased", [
    sa + (unbiased,) for sa in shapes_and_axis for unbiased in [True, False]   
])
def test_std(shape, axis, unbiased):
    a_milli, a_torch = create_randn_pair(shape)
    
    std_milli = a_milli.std(axis=axis, unbiased=unbiased)
    std_torch = a_torch.std(dim=axis, unbiased=unbiased)
    
    np.testing.assert_allclose(
        std_milli.data, std_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    std_milli.backward()
    std_torch.backward(torch.ones_like(std_torch))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
    
if __name__ == "__main__":
    test_std((2, 3, 4), -1, True)
    