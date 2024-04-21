from milligrad.tensor import Tensor
from tests.utils import create_randn_pair

import pytest
import torch
import numpy as np

shapes = [
    (2,),
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 5),
]

@pytest.mark.parametrize("shape", shapes)
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
    
@pytest.mark.parametrize("shape", shapes)
def test_sub(shape): # tests __neg__ and __add__
    a_milli, a_torch = create_randn_pair(shape)
    b_milli, b_torch = create_randn_pair(shape)
    
    milli_sub = a_milli - b_milli
    torch_sub = a_torch - b_torch
    np.testing.assert_allclose(
        milli_sub.data, torch_sub.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    # trivial backwards pass
    milli_sub.backward()
    torch_sub.backward(torch.ones_like(torch_sub))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    np.testing.assert_allclose(
        b_milli.grad, b_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
@pytest.mark.parametrize("shape", shapes)
def test_mul(shape):
    a_milli, a_torch = create_randn_pair(shape)
    b_milli, b_torch = create_randn_pair(shape)
    
    milli_mul = a_milli * b_milli
    torch_mul = a_torch * b_torch
    np.testing.assert_allclose(
        milli_mul.data, torch_mul.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    # trivial backwards pass
    milli_mul.backward()
    torch_mul.backward(torch.ones_like(torch_mul))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    np.testing.assert_allclose(
        b_milli.grad, b_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
@pytest.mark.parametrize("shape", shapes)
def test_div(shape):
    a_milli, a_torch = create_randn_pair(shape)
    b_milli, b_torch = create_randn_pair(shape)
    
    milli_div = a_milli / b_milli
    torch_div = a_torch / b_torch
    np.testing.assert_allclose(
        milli_div.data, torch_div.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    # trivial backwards pass
    milli_div.backward()
    torch_div.backward(torch.ones_like(torch_div))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=5e-4
    )
    np.testing.assert_allclose(
        b_milli.grad, b_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=5e-4
    )
    
@pytest.mark.parametrize("shape, pow", list(zip(shapes, [2.0, 1.0, 3.0, 2.0])))
def test_pow(shape, pow):
    a_milli, a_torch = create_randn_pair(shape)
    
    milli_pow = a_milli ** pow
    torch_pow = a_torch ** pow
    np.testing.assert_allclose(
        milli_pow.data, torch_pow.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    milli_pow.backward()
    torch_pow.backward(torch.ones_like(torch_pow))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
@pytest.mark.parametrize("shape", shapes)
def test_log(shape):
    __import__("warnings").filterwarnings("ignore", message="invalid value")
    a_milli, a_torch = create_randn_pair(shape)
    
    milli_log = a_milli.log()
    torch_log = a_torch.log()
    np.testing.assert_allclose(
        milli_log.data, torch_log.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-5
    )
    milli_log.backward()
    torch_log.backward(torch.ones_like(torch_log))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-5
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
    for shape in shapes:
        test_div(shape)