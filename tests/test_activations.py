from milligrad.tensor import Tensor
from tests.utils import create_randn_pair

import pytest
import torch
import torch.nn.functional as F
import numpy as np

shapes = [
    (2,),
    (2, 3),
    (2, 3, 4),
]

@pytest.mark.parametrize("shape", shapes)
def test_relu(shape):
    a_milli, a_torch = create_randn_pair(shape)
    
    a_milli_relu = a_milli.relu()
    a_torch_relu = F.relu(a_torch)
    
    np.testing.assert_allclose(
        a_milli_relu.data, a_torch_relu.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    a_milli_relu.backward()
    a_torch_relu.backward(torch.ones_like(a_torch_relu))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
@pytest.mark.parametrize("shape", shapes)
def test_tanh(shape):
    a_milli, a_torch = create_randn_pair(shape)
    
    a_milli_tanh = a_milli.tanh()
    a_torch_tanh = torch.tanh(a_torch)
    
    np.testing.assert_allclose(
        a_milli_tanh.data, a_torch_tanh.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    a_milli_tanh.backward()
    a_torch_tanh.backward(torch.ones_like(a_torch_tanh))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
def test_sigmoid():
    a_milli, a_torch = create_randn_pair((2,3))
    
    a_milli_sigmoid = a_milli.sigmoid()
    a_torch_sigmoid = torch.sigmoid(a_torch)
    
    np.testing.assert_allclose(
        a_milli_sigmoid.data, a_torch_sigmoid.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    a_milli_sigmoid.backward()
    a_torch_sigmoid.backward(torch.ones_like(a_torch_sigmoid))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
def test_softmax():
    a_milli, a_torch =  create_randn_pair((2,3))
    
    a_milli_softmax = a_milli.softmax()
    a_torch_softmax = F.softmax(a_torch, dim=-1)
    
    np.testing.assert_allclose(
        a_milli_softmax.data, a_torch_softmax.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    a_milli_softmax.backward()
    a_torch_softmax.backward(torch.ones_like(a_torch_softmax))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )

def test_log_softmax():
    a_milligrad, a_torch =  create_randn_pair((2,3))

    log_softmax_milligrad = a_milligrad.log_softmax()
    log_softmax_torch = F.log_softmax(a_torch, dim=-1)

    np.testing.assert_allclose(
        log_softmax_milligrad.data, log_softmax_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )

    log_softmax_milligrad.backward()
    log_softmax_torch.backward(torch.ones_like(log_softmax_torch))
    
    np.testing.assert_allclose(
        a_milligrad.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )