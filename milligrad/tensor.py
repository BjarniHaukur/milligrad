# Heavily inspired by Andrej Karpathy's micrograd and George Hotz' tinygrad
# automatic differentiation with a numpy backend
import numpy as np


def topological_sort(tensor:"Tensor")->list["Tensor"]:
    visited = set()
    stack = []
    
    def _topological_sort(tensor:"Tensor"):
        if tensor not in visited:
            visited.add(tensor)
            for child in tensor._prev:
                _topological_sort(child)
            stack.append(tensor)
    
    _topological_sort(tensor)
    return stack

class Tensor:
    _no_grad = False
    class no_grad:
        def __enter__(self): self.prev_no_grad, Tensor._no_grad = Tensor._no_grad, True 
        def __exit__(self, *args): Tensor._no_grad = self.prev_no_grad

    def __init__(self, data:np.ndarray|list, _children:tuple["Tensor"]=(), name=""):
        self.data = np.array(data)
        self.grad = np.zeros_like(data)
        
        self._backward = lambda: None # the closure, added by operators
        # "ordered set" (dict keys are ordered in python 3.7+)
        self._prev = dict.fromkeys(reversed(_children)) # reversed to backpropagate in the right order
        self.name = name # nice to have
        
    @classmethod
    def zeros(cls, *shape:int, name="")->"Tensor": return cls(np.zeros(shape), name=name)
    @classmethod
    def ones(cls, *shape:int, name="")->"Tensor": return cls(np.ones(shape), name=name)
    @classmethod
    def randn(cls, *shape:int, name="")->"Tensor": return cls(np.random.randn(*shape), name=name)
    @classmethod
    def xavier(cls, n_in:int, n_out:int, name="")->"Tensor":
        bound = np.sqrt(6/(n_in + n_out))
        return cls(np.random.uniform(-bound, bound, (n_in, n_out)), name=name)
    
    def __add__(self, other:"Tensor")->"Tensor":
        out = Tensor(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += out.grad
            broadcasted = self.shape != other.shape # not pretty, assumes broadcasting over batch
            other.grad += np.sum(out.grad, axis=0) if broadcasted else out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def __matmul__(self, other:"Tensor")->"Tensor":
        out = Tensor(self.data @ other.data, (self, other), "@")
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def __pow__(self, power:int|float)->"Tensor":
        out = Tensor(self.data**power, (self,), f"**{power}")
        
        def _backward():
            self.grad += power * self.data**(power-1) * out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def backward(self):
        assert not Tensor._no_grad, "No gradient tracking when in no_grad mode"
        self.grad = np.ones_like(self.data) # dL/dL = 1
        
        for tensor in reversed(topological_sort(self)):
            # the _backward functions keep copies of their data in the closure
            tensor._backward()
            
    def __neg__(self)->"Tensor":
        out = Tensor(-self.data, (self,), "-")
        
        def _backward():
            self.grad += -out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def __sub__(self, other:"Tensor")->"Tensor":
        return self + (-other)
    
    def sum(self, axis:int=-1)->"Tensor":
        out = Tensor(self.data.sum(axis), (self,), "sum")
        
        def _backward():
            self.grad += np.expand_dims(out.grad, axis) # broadcast the gradient
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def relu(self)->"Tensor":
        out = Tensor(np.maximum(self.data, 0), (self,), "relu")
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def tanh(self)->"Tensor":
        out = Tensor(np.tanh(self.data), (self,), "tanh")
        
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def sigmoid(self)->"Tensor":
        out = Tensor(1/(1 + np.exp(-self.data)), (self,), "sigmoid")
        
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def softmax(self)->"Tensor":
        e = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True)) # for numerical stability
        out = Tensor(e / np.sum(e, axis=-1, keepdims=True), (self,), "softmax")
        
        def _backward():
            jacobian = np.zeros((out.data.size, out.data.size))
            flat_out = out.data.flatten()
            jacobian = flat_out[:,None] * (np.eye(out.data.size) - flat_out[None,:])
            self.grad += (jacobian.T @ out.grad.flatten()).reshape(self.data.shape)
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def log_softmax(self)->"Tensor":
        e = self.data - np.max(self.data, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(e), axis=-1, keepdims=True))
        out = Tensor(e - log_sum_exp, (self,), "log_softmax")
        
        def _backward():
            jacobian = np.eye(out.data.size) - np.exp(out.data).flatten()
            self.grad += (jacobian.T @ out.grad.flatten()).reshape(self.data.shape)
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def nll_loss(self, target:"Tensor")->"Tensor":
        assert self.shape == target.shape, "Input and target shapes must match"
        out = Tensor(-self.data[np.arange(len(self.data)), target.data].mean(), (self, target), "nll_loss")
        
        def _backward():
            self.grad += np.zeros_like(self.data)
            self.grad[np.arange(len(self.data)), target.data] = -1/len(self.data)
            
        if not Tensor._no_grad: self._backward = _backward
        return out
        
    def categorical_cross_entropy(self, target:"Tensor")->"Tensor":
        assert self.shape == target.shape, "Input and target shapes must match"
        out = Tensor(-np.sum(target.data * self.data, axis=-1), (self, target), "categorical_cross_entropy")
        
        def _backward():
            self.grad += -target.data
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    @property
    def T(self)->"Tensor":
        out = Tensor(self.data.T, (self,), "T")
        
        def _backward():
            self.grad += out.grad.T 

        if not Tensor._no_grad: self._backward = _backward
        return out
            
    def __repr__(self):
        return self.data.__repr__() # use the numpy repr
        
    @property
    def shape(self)->tuple[int]:
        return self.data.shape