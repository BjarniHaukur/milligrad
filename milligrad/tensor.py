# Heavily inspired by Andrej Karpathy's micrograd and George Hotz' tinygrad
# automatic differentiation with a numpy backend
from __future__ import annotations

import numpy as np


def topological_sort(tensor:Tensor)->list[Tensor]:
    visited = set()
    stack = []
    
    def _topological_sort(tensor:Tensor):
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

    def __init__(self, data:np.ndarray|list, _children:tuple[Tensor]=(), name=""):
        self.data = np.array(data)
        self.grad = np.zeros_like(data)
        self.name = name # nice to have
        
        self._backward = lambda: None # the closure, added by operators
        # "ordered set" (dict keys are ordered in python 3.7+)
        # reversed to backpropagate in the right order
        # required to avoid circular references (e.g. a + a)
        self._prev = dict.fromkeys(reversed(_children)).keys() 
        
    @classmethod
    def zeros(cls, *shape:int, name:str=""): return cls(np.zeros(shape), name=name)
    @classmethod
    def ones(cls, *shape:int, name:str=""): return cls(np.ones(shape), name=name)
    @classmethod
    def randn(cls, *shape:int, name:str=""): return cls(np.random.randn(*shape), name=name)
    @classmethod
    def xavier(cls, n_in:int, n_out:int, name:str=""):
        bound = np.sqrt(6/(n_in + n_out))
        return cls(np.random.uniform(-bound, bound, (n_in, n_out)), name=name)
    
    def backward(self):
        assert not Tensor._no_grad, "No gradient tracking when in no_grad mode"
        self.grad = np.ones_like(self.data) # dL/dL = 1
        
        for tensor in reversed(topological_sort(self)):
            # the _backward functions keep copies of their data in the closure
            tensor._backward()
            
    ###################################################################################
    ##### The following operations perform all the necessary gradient bookkeeping #####
    ###################################################################################
    
    def __add__(self, other:Tensor|int|float)->Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += out.grad
            broadcasted = self.shape != other.shape # not pretty, assumes broadcasting over batch
            other.grad += np.sum(out.grad, axis=-1, keepdims=True) if broadcasted else out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def __neg__(self)->Tensor:
        out = Tensor(-self.data, (self,), "-")
        
        def _backward():
            self.grad += -out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    # the * operator is element-wise multiplication
    def __mul__(self, other:Tensor|int|float)->Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.grad += other.data * out.grad
            broadcasted = self.shape != other.shape # assumes broadcasting over last axis
            other.grad += np.sum(self.data * out.grad, axis=-1, keepdims=True) if broadcasted else self.data * out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    # the @ operator is matrix multiplication
    def __matmul__(self, other:Tensor)->Tensor:
        out = Tensor(self.data @ other.data, (self, other), "@")
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    # the ** operator is element-wise power
    def __pow__(self, power:int|float)->Tensor:
        out = Tensor(self.data**power, (self,), f"**{power}")
        
        def _backward():
            self.grad += power * self.data**(power-1) * out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def pow(self, base:int|float=np.e)->Tensor:
        out = Tensor(base**self.data, (self,), f"{base}**")
        
        def _backward():
            self.grad += np.log(base) * base**self.data * out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def sum(self, axis:int=-1)->Tensor:
        out = Tensor(self.data.sum(axis), (self,), "sum")
        
        def _backward():
            self.grad += np.expand_dims(out.grad, axis) # broadcast the gradient
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def mean(self, axis:int=-1)->Tensor:
        out = Tensor(self.data.mean(axis), (self,), "mean")
        
        def _backward():
            self.grad += np.expand_dims(out.grad, axis) / self.data.shape[axis]
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def std(self, axis:int=-1)->Tensor:
        mean = self.data.mean(axis, keepdims=True)
        std_dev = np.sqrt(((self.data - mean)**2).mean(axis, keepdims=True))
        out = Tensor(std_dev, (self,), "std")
        
        def _backward():
            N = self.data.shape[axis]
            self.grad += np.expand_dims(out.grad, axis) * (self.data - mean) / (N * std_dev)
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def max(self, axis:int=-1)->Tensor:
        out = Tensor(self.data.max(axis=axis, keepdims=True), (self,), "max")
        
        def _backward():
            self.grad += (self.data == out.data) * out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def relu(self)->Tensor:
        out = Tensor(np.maximum(self.data, 0), (self,), "relu")
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    # could be handled via a bunch of other ops, but the derivative of tanh is well-known
    def tanh(self)->Tensor:
        out = Tensor(np.tanh(self.data), (self,), "tanh")
        
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    # same for sigmoid
    def sigmoid(self)->Tensor:
        out = Tensor(1/(1 + np.exp(-self.data)), (self,), "sigmoid")
        
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    ############################ Shape-changing operations ############################
    
    @property
    def T(self)->Tensor:
        out = Tensor(self.data.T, (self,), "T")
        
        def _backward():
            self.grad += out.grad.T 

        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def reshape(self, *shape:int)->Tensor:
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")
        
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    ###################################################################################
    ############## The following functions simply apply other functions ###############
    ###################################################################################
    
    def __sub__(self, other:Tensor|int|float)->Tensor: return self + (-other)
    def __radd__(self, other:int|float)->Tensor: return self + other
    def __rsub__(self, other:int|float)->Tensor: return other + (-self)
    def __rmul__(self, other:int|float)->Tensor: return self * other
    def __truediv__(self, other:Tensor)->Tensor: return self * other**-1
    
    def min(self, axis:int=-1)->Tensor:
        return -(-self).max(axis)
    
    def softmax(self, axis:int=-1)->Tensor:
        shifted_exp = (self - self.max(axis)).pow()
        return shifted_exp / shifted_exp.sum(axis).reshape(-1, 1)
        
    
        pass
        ## for numerical stability we shift the data by its maximum value
        # shift_exp = np.exp(self.data - self.data.max(axis=-1, keepdims=True))
        # out = Tensor(shift_exp / shift_exp.sum(axis=-1, keepdims=True), (self,), "softmax")
        
        # def _backward():
        #     # d(softmax(x))/dx = softmax(x) * (I - softmax(x)^)
        #     softmax_output = out.data
        #     jacobian = softmax_output * (np.eye(softmax_output.shape[-1]) - softmax_output.reshape(-1, 1))
        #     self.grad += np.dot(out.grad.reshape(1, -1), jacobian).reshape(self.data.shape)
            
        # if not Tensor._no_grad: self._backward = _backward
        # return out
    
    def log_softmax(self)->Tensor:
        pass
        # softmax = self.softmax()
        # out = Tensor(np.log(softmax.data), (self,), "log_softmax")
        
        # def _backward():
        #     self.grad += (softmax.data - 1) * out.grad
            
        # if not Tensor._no_grad: self._backward = _backward
        # return out
        
    def categorical_crossentropy(self, target:Tensor)->Tensor:
        pass
        # return (-target * self.log_softmax()).sum().mean()
        
    def mse(self, target:Tensor)->Tensor:
        pass
        # return ((self - target)**2).sum().mean()
            
    def __repr__(self):
        return self.data.__repr__() # use the numpy repr
        
    @property
    def shape(self)->tuple[int]:
        return self.data.shape