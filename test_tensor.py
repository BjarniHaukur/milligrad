from milligrad.tensor import Tensor

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
    

def test_perceptron():
    x_np = np.random.randn(5, 3)
    W_np, b_np = np.random.randn(3, 2), np.random.randn(1, 2)
    y_hat_np = np.maximum(x_np @ W_np + b_np, 0)
    
    x = Tensor(x_np)
    W, b = Tensor(W_np), Tensor(b_np)
    y_hat = (x @ W + b).relu()
    
    np.testing.assert_allclose(y_hat.data, y_hat_np)
    

def test_perceptron_grad():
    x_np, y_np = np.random.randn(5, 3), np.random.randn(5, 2)
    W_np, b_np = np.random.randn(3, 2), np.random.randn(2)
    y_hat_np = x_np @ W_np + b_np
    L_np = ((y_np - y_hat_np)**2).sum()
    
    W_grad_true = -2 * x_np.T @ (y_np - y_hat_np)
    b_grad_true = -2 * (y_np - y_hat_np).sum(axis=0)
    
    x, y = Tensor(x_np, name="input"), Tensor(y_np, name="target")
    W, b = Tensor(W_np, name="W"), Tensor(b_np, name="b")
    y_hat = x @ W + b
    diff = y - y_hat
    squared = diff ** 2
    # L = ((y - y_hat)**2).sum().sum()
    L = squared.sum().sum()
    L.backward()
    
    np.testing.assert_allclose(L.data, L_np)
    np.testing.assert_allclose(W.grad, W_grad_true)
    np.testing.assert_allclose(b.grad, b_grad_true)
    
    
    

    