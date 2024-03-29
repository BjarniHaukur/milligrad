from milligrad.tensor import Tensor

import torch
import numpy as np

def test_add():
    a_np = np.random.randn(3, 4)
    b_np = np.random.randn(3, 4)
    
    a = Tensor(a_np)
    b = Tensor(b_np)
    
    np.testing.assert_allclose((a + b).data, a_np + b_np)

def test_matmul():
    a_np = np.random.randn(3, 4)
    b_np = np.random.randn(4, 5)
    
    a = Tensor(a_np)
    b = Tensor(b_np)
    
    np.testing.assert_allclose((a @ b).data, np.matmul(a_np, b_np))
    
def test_transpose():
    a_np = np.random.randn(3, 4)
    
    a = Tensor(a_np)
    
    np.testing.assert_allclose(a.T.data, a_np.T)
    
def test_sum():
    a_np = np.random.randn(3, 4)
    
    a = Tensor(a_np)
    
    np.testing.assert_allclose(a.sum(0).data, a_np.sum(axis=0))
    np.testing.assert_allclose(a.sum(1).data, a_np.sum(axis=1))
    
def test_mean():
    # Test case 1: Mean along the last axis
    a_np = np.random.randn(3, 4)
    a_torch = torch.tensor(a_np, requires_grad=True)
    a_milligrad = Tensor(a_np)

    mean_torch = a_torch.mean(dim=-1)
    mean_milligrad = a_milligrad.mean(axis=-1)

    np.testing.assert_allclose(
        mean_milligrad.data, mean_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )

    mean_torch.backward(torch.ones_like(mean_torch))
    grad_pytorch = a_torch.grad.numpy()

    mean_milligrad.backward()
    grad_milligrad = a_milligrad.grad

    np.testing.assert_allclose(
        grad_milligrad, grad_pytorch,
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )

    # Test case 2: Mean along a specific axis
    b_np = np.random.randn(2, 3, 4)
    b_torch = torch.tensor(b_np, requires_grad=True)
    b_milligrad = Tensor(b_np)

    mean_torch = b_torch.mean(dim=1)
    mean_milligrad = b_milligrad.mean(axis=1)

    np.testing.assert_allclose(
        mean_milligrad.data, mean_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )

    mean_torch.backward(torch.ones_like(mean_torch))
    grad_pytorch = b_torch.grad.numpy()

    mean_milligrad.backward()
    grad_milligrad = b_milligrad.grad

    np.testing.assert_allclose(
        grad_milligrad, grad_pytorch,
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
def test_softmax():
    a_np = np.random.randn(3, 4)

    a_torch = torch.tensor(a_np, dtype=torch.float32, requires_grad=True)
    a_milligrad = Tensor(a_np)

    softmax_torch = torch.softmax(a_torch, dim=-1)
    softmax_milligrad = a_milligrad.softmax()

    np.testing.assert_allclose(
        softmax_milligrad.data, softmax_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )

    softmax_torch.backward(torch.ones_like(softmax_torch))
    grad_pytorch = a_torch.grad.numpy()

    # when you call backward in milligrad, the gradients are initialized to ones
    softmax_milligrad.backward()
    grad_milligrad = a_milligrad.grad
    
    # Compare the gradients from the backward pass
    np.testing.assert_allclose(
        grad_milligrad, grad_pytorch,
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )
    
def test_log_softmax():
    a_np = np.random.randn(3, 4)

    a_torch = torch.tensor(a_np, dtype=torch.float32, requires_grad=True)
    a_milligrad = Tensor(a_np)

    log_softmax_torch = torch.nn.functional.log_softmax(a_torch, dim=-1)
    log_softmax_milligrad = a_milligrad.log_softmax()

    np.testing.assert_allclose(
        log_softmax_milligrad.data, log_softmax_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )

    log_softmax_torch.backward(torch.ones_like(log_softmax_torch))
    grad_pytorch = a_torch.grad.numpy()

    # when you call backward in milligrad, the gradients are initialized to ones
    log_softmax_milligrad.backward()
    grad_milligrad = a_milligrad.grad
    
    # Compare the gradients from the backward pass
    np.testing.assert_allclose(
        grad_milligrad, grad_pytorch,
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )

def test_cross_entropy():
    a_np = np.random.randn(3, 4)
    y_np = np.eye(4)[np.random.randint(0, 4, size=(3))]
    
    a_torch = torch.tensor(a_np, dtype=torch.float32, requires_grad=True)
    y_torch = torch.tensor(y_np, dtype=torch.float32)
    
    a_milligrad = Tensor(a_np)
    y_milligrad = Tensor(y_np)
    
    loss_torch = torch.nn.functional.cross_entropy(a_torch, y_torch.argmax(dim=-1))
    loss_milligrad = a_milligrad.cross_entropy(y_milligrad)
    
    np.testing.assert_allclose(
        loss_milligrad.data, loss_torch.detach().numpy(),
        err_msg="Forward pass mismatch", atol=1e-6
    )
    
    loss_torch.backward()
    grad_pytorch = a_torch.grad.numpy()
    
    loss_milligrad.backward()
    grad_milligrad = a_milligrad.grad
    
    # Compare the gradients from the backward pass
    np.testing.assert_allclose(
        grad_milligrad, grad_pytorch,
        err_msg="Backward pass gradient mismatch", atol=1e-6
    )    

def test_perceptron():
    x_np = np.random.randn(5, 3)
    w_np, b_np = np.random.randn(3, 2), np.random.randn(1, 2)
    y_hat_np = np.maximum(x_np @ w_np + b_np, 0)
    
    x = Tensor(x_np)
    W, b = Tensor(w_np), Tensor(b_np)
    y_hat = (x @ W + b).relu()
    
    np.testing.assert_allclose(y_hat.data, y_hat_np)

def test_perceptron_grad():
    x_np, y_np = np.random.randn(5, 3), np.random.randn(5, 2)
    w_np, b_np = np.random.randn(3, 2), np.random.randn(2)
    y_hat_np = x_np @ w_np + b_np
    l_np = ((y_np - y_hat_np)**2).sum()
    
    w_grad_true = -2 * x_np.T @ (y_np - y_hat_np)
    b_grad_true = -2 * (y_np - y_hat_np).sum(axis=0)
    
    x, y = Tensor(x_np), Tensor(y_np)
    w, b = Tensor(w_np), Tensor(b_np)
    y_hat = x @ w + b
    l = ((y - y_hat)**2).sum().sum()
    l.backward()
    
    np.testing.assert_allclose(l.data, l_np)
    np.testing.assert_allclose(w.grad, w_grad_true)
    np.testing.assert_allclose(b.grad, b_grad_true)
    
    
if __name__ == "__main__":
    test_softmax()
    test_log_softmax()
    test_cross_entropy()