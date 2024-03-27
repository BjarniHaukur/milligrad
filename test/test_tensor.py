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
    
def test_softmax():
    a_np = np.random.randn(3, 4)
    a = Tensor(a_np)
    
    softmax_np = np.exp(a_np - np.max(a_np, axis=-1, keepdims=True))
    softmax_np /= np.sum(softmax_np, axis=-1, keepdims=True)
    
    np.testing.assert_allclose(a.softmax().data, softmax_np)

def test_softmax_grad():
    a_np = np.random.randn(3, 4)
    a = Tensor(a_np)
    
    softmax = a.softmax()
    jacobian_np = np.zeros((softmax.data.size, softmax.data.size))
    flat_softmax = softmax.data.flatten()
    jacobian_np = flat_softmax[:,None] * (np.eye(softmax.data.size) - flat_softmax[None,:])
    
    softmax.backward()
    
    np.testing.assert_allclose(a.grad, (jacobian_np.T @ np.ones_like(softmax.data).flatten()).reshape(a.data.shape))

def test_categorical_cross_entropy():
    a_np = np.random.randn(3, 4)
    b_np = np.eye(4)[np.random.randint(0, 4, size=3)]
    
    a = Tensor(a_np).log_softmax()
    b = Tensor(b_np)
    
    ce_np = -np.sum(b_np * a.data, axis=-1)
    
    np.testing.assert_allclose(a.categorical_cross_entropy(b).data, ce_np)

def test_categorical_cross_entropy_grad():
    a_np = np.random.randn(3, 4)
    b_np = np.eye(4)[np.random.randint(0, 4, size=3)]
    
    a = Tensor(a_np).log_softmax()
    b = Tensor(b_np)
    
    ce = a.categorical_cross_entropy(b)
    ce.backward()
    
    np.testing.assert_allclose(a.grad, -b.data)

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
    
    x, y = Tensor(x_np, name="input"), Tensor(y_np, name="target")
    W, b = Tensor(w_np, name="W"), Tensor(b_np, name="b")
    y_hat = x @ W + b
    l = ((y - y_hat)**2).sum().sum()
    l.backward()
    
    np.testing.assert_allclose(l.data, l_np)
    np.testing.assert_allclose(W.grad, W_grad_true)
    np.testing.assert_allclose(b.grad, b_grad_true)
    
    
    

    