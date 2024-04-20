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
def test_add(shape):    
    a_milli, a_torch = create_randn_pair(shape)
    b_milli, b_torch = create_randn_pair(shape)
    
    milli_sum = a_milli + b_milli
    torch_sum = a_torch + b_torch
    np.testing.assert_allclose(
        milli_sum.data, torch_sum.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    # trivial backwards pass
    milli_sum.backward()
    torch_sum.backward(torch.ones_like(torch_sum))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    np.testing.assert_allclose(
        b_milli.grad, b_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
@pytest.mark.parametrize("shape_a, shape_b", [
    ((1,), (1,)), # -> (1,)
    ((2,), (2,)), # -> (2,)
    ((2, 3), (3,)), # -> (2,)
    ((2,), (2, 3)), # -> (3,)
    ((2, 3), (3, 4)), # -> (2, 4)
    ((2, 3, 4), (4,)), # -> (2, 3,)
    ((2, 3, 4), (4, 5)), # -> (2, 3, 5)
    ((2, 3, 4), (2, 4, 5)), # -> (2, 3, 5)
    ((3,), (2, 3, 4)), # -> (2, 4)
    ((2, 3), (2, 3, 4)), # -> (2, 2, 4) because of broadcasting 
])
def test_matmul(shape_a, shape_b):
    a_milli, a_torch = create_randn_pair(shape_a)
    b_milli, b_torch = create_randn_pair(shape_b)
    
    milli_matmul = a_milli @ b_milli
    torch_matmul = a_torch @ b_torch
    
    np.testing.assert_allclose(
        milli_matmul.data, torch_matmul.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    milli_matmul.backward()
    torch_matmul.backward(torch.ones_like(torch_matmul))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    np.testing.assert_allclose(
        b_milli.grad, b_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )

if __name__ == "__main__":
    test_matmul((2, 3), (2, 3, 4))