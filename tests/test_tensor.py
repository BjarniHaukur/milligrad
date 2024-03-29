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
        err_msg="Backward pass gradient mismatch", atol=1e-7
    )


def test_categorical_cross_entropy():
    pass

def test_categorical_cross_entropy_grad():
    pass

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
    
    W_grad_true = -2 * x_np.T @ (y_np - y_hat_np)
    b_grad_true = -2 * (y_np - y_hat_np).sum(axis=0)
    
    x, y = Tensor(x_np), Tensor(y_np)
    W, b = Tensor(w_np), Tensor(b_np)
    y_hat = x @ W + b
    l = ((y - y_hat)**2).sum().sum()
    l.backward()
    
    np.testing.assert_allclose(l.data, l_np)
    np.testing.assert_allclose(W.grad, W_grad_true)
    np.testing.assert_allclose(b.grad, b_grad_true)
    
    
    

    
if __name__ == "__main__":
    test_softmax()