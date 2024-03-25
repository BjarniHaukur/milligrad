# Heavily inspired by Andrej Karpathy's micrograd and George Hotz' tinygrad
# adds automatic differentiation to numpy
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
        def __enter__(self):
            self.previous_no_grad, Tensor._no_grad = Tensor._no_grad, True 
            return self
        def __exit__(self, *args): Tensor._no_grad = self.previous_no_grad

    def __init__(self, data:np.ndarray|list, _children:tuple["Tensor"]=(), name=""):
        self.data = np.array(data)
        self.grad = np.zeros_like(data)
        
        self._backward = lambda: None # the closure, added by operators
        # "ordered set" (dict keys are ordered in python 3.7+)
        self._prev = dict.fromkeys(reversed(_children)) # reversed to backpropagate in the right order
        self.name = name # nice to have
        
    @classmethod
    def zeros(cls, *shape:int)->"Tensor": return cls(np.zeros(shape))
    @classmethod
    def ones(cls, *shape:int)->"Tensor": return cls(np.ones(shape))
    @classmethod
    def randn(cls, *shape:int)->"Tensor": return cls(np.random.randn(*shape))
    @classmethod
    def xavier(cls, n_in:int, n_out:int)->"Tensor":
        bound = np.sqrt(6/(n_in + n_out))
        return cls(np.random.uniform(-bound, bound, (n_in, n_out)))
    
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