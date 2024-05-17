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


@pytest.mark.parametrize("tensor_shapes, n_tensors, axis", [
    ((2, 3), 1, 0),
    ((2, 3), 1, 1),
    ((3, 4), 4, 0),
    ((3, 4), 4, 1),
    ((2, 2, 2), 4, 0),
    ((2, 2, 2), 4, 1),
    ((2, 2, 2), 4, 2),
])
def test_stack(tensor_shapes:tuple[int], n_tensors:int, axis:int):
    tensors_milli, tensors_torch = map(list, zip(*[create_randn_pair(tensor_shapes) for _ in range(n_tensors)]))

    stacked_milli = Tensor.stack(tensors_milli, axis=axis)
    stacked_torch = torch.stack(tensors_torch, dim=axis)

    np.testing.assert_allclose(
        stacked_milli.data, stacked_torch.detach().numpy(),
        err_msg="Forward pass mismatch in stack", atol=1e-5
    )

    stacked_milli.std(unbiased=False).backward() # to get non-trivial values in the gradient
    stacked_torch.std(unbiased=False).backward()

    for i, (milli_tensor, torch_tensor) in enumerate(zip(tensors_milli, tensors_torch)):
        np.testing.assert_allclose(
            milli_tensor.grad, torch_tensor.grad.numpy(),
            err_msg=f"Backward pass gradient mismatch in stack item {i=}", atol=1e-5
        )

@pytest.mark.parametrize("shape, index", [
    ((2, 3), 0),
    ((2, 3), slice(0, 2)),
    ((2, 3, 4), (slice(0, 2), 1)),
    ((2, 3, 4, 5), (slice(None), slice(None), slice(1, 3))),
    ((2, 3, 4, 5), (1, slice(None), 2)),
])
def test_getitem(shape, index):
    a_milli, a_torch = create_randn_pair(shape)
    
    a_milli_sub = a_milli[index]
    a_torch_sub = a_torch[index]
    
    np.testing.assert_allclose(
        a_milli_sub.data, a_torch_sub.detach().numpy(),
        err_msg="Forward pass mismatch in getitem", atol=1e-6
    )
    
    a_milli_sub.backward()
    a_torch_sub.backward(torch.ones_like(a_torch_sub))
    
    np.testing.assert_allclose(
        a_milli.grad, a_torch.grad.numpy(),
        err_msg="Backward pass gradient mismatch in getitem", atol=1e-6
    )


if __name__ == "__main__":
    test_stack((3, 4), 4, 0)