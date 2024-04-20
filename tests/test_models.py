from milligrad.tensor import Tensor

import numpy as np

def test_perceptron():
    x_np = np.random.randn(5, 3)
    w_np, b_np = np.random.randn(3, 2), np.random.randn(2)
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
    l = ((y - y_hat)**2).sum()
    l.backward()
    
    np.testing.assert_allclose(l.data, l_np)
    np.testing.assert_allclose(w.grad, w_grad_true)
    np.testing.assert_allclose(b.grad, b_grad_true)