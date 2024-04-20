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

def broadcast_to(grad:np.ndarray|np.float32, shape:tuple[int])->np.ndarray:
    if grad.ndim == 0: 
        return np.broadcast_to(grad, shape)
    if grad.ndim < len(shape):
        new_shape = [i if i in grad.shape else 1 for i in shape] # works since dimensions are dropped inplace
        return np.broadcast_to(grad.reshape(new_shape), shape)
    if grad.ndim > len(shape):
        axis = tuple(i for i, s in enumerate(grad.shape) if s not in shape)
        return grad.sum(axis=axis) 
    return grad
    
class Tensor:
    _no_grad = False
    class no_grad:
        def __enter__(self): self.prev_no_grad, Tensor._no_grad = Tensor._no_grad, True 
        def __exit__(self, *args): Tensor._no_grad = self.prev_no_grad

    def __init__(self, data:np.ndarray|list|int|float, _children:tuple[Tensor]=(), _grad_fn:str=""):
        # sometimes isinstance fails in notebooks when changing the class definition
        # bails out here if data is e.g. Tensor ^^^
        assert isinstance(data, (np.ndarray, list, int, float)), f"Invalid data type {type(data)}"
        self.data = np.array(data)
        self.grad = np.zeros_like(data)
        
        self._backward = lambda: None # a closure, added by operators
        self._prev = dict.fromkeys(reversed(_children)).keys() # "ordered set" of children
        self._grad_fn = _grad_fn # nice to have
        
    @classmethod
    def zeros(cls, *shape:int): return cls(np.zeros(shape))
    @classmethod
    def ones(cls, *shape:int): return cls(np.ones(shape))
    @classmethod
    def randn(cls, *shape:int): return cls(np.random.randn(*shape))
    @classmethod
    def xavier(cls, n_in:int, n_out:int):
        bound = np.sqrt(6/(n_in + n_out))
        return cls(np.random.uniform(-bound, bound, (n_in, n_out)))
    
    def backward(self):
        assert not Tensor._no_grad, "No gradient tracking when in no_grad mode"
        self.grad = np.ones_like(self.data) # dL/dL = 1
        
        for tensor in reversed(topological_sort(self)):
            tensor._backward() # relevant data kept in these closures
            tensor._backward = lambda: None # free memory, drop closures
            
    ###################################################################################
    ##### The following operations perform all the necessary gradient bookkeeping #####
    ###################################################################################
    # note that we use += instead of assignments in the _backward since the same tensor
    # can be used multiple times in the computation graph
    
    def __add__(self, other:Tensor|int|float)->Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += broadcast_to(out.grad, self.shape)
            other.grad += broadcast_to(out.grad, other.shape)
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    # the * operator is element-wise multiplication
    def __mul__(self, other:Tensor|int|float)->Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.grad += broadcast_to(out.grad * other.data, self.shape)
            other.grad += broadcast_to(out.grad * self.data, other.shape)
        
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    # the @ operator is matrix multiplication
    def __matmul__(self, other: Tensor) -> Tensor:
        a, b = self.data, other.data
        if a_vec := a.ndim == 1: a = a[None] # treat a as a row vector
        if b_vec := b.ndim == 1: b = b[..., None] # treat b as a column vector
        
        # add unary dimensions so that a and b have the same number of dimensions, summed away later
        a_diff, b_diff = tuple(range(b.ndim - a.ndim)), tuple(range(a.ndim - b.ndim))
        a = a.reshape((1,) * len(a_diff) + a.shape)
        b = b.reshape((1,) * len(b_diff) + b.shape)
        
        c = a @ b # we use the shape of this later to figure out the gradient
        out = Tensor(c.squeeze() if a_vec or b_vec else c, (self, other), "@")
        
        def _backward():
            dc = out.grad.reshape(c.shape) # adds back the unary dimensions
            self.grad += (dc @ np.moveaxis(b, -1, -2)).sum(a_diff).sum((0,) if a_vec else ())
            other.grad += (np.moveaxis(a, -1, -2) @ dc).sum(b_diff).sum((-1,) if b_vec else ())
        
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def conv1d(self, kernels:Tensor, padding:int=0)->Tensor:
        assert self.data.ndim == 3, f"Input tensor must be batched 2d i.e. 3d but got {self.data.ndim=}"
        assert kernels.data.ndim == 3, f"Expected (c_in, kernel_size, c_out) but got {kernels.shape=} instead"
        assert self.shape[1] == kernels.shape[0], f"Input channel mismatch"
        
        B, C_in, W_in = self.shape
        C_in, K, C_out = kernels.shape
        W_out = W_in - K + 1 + 2*padding
    
        pad = np.pad(self.data, ((0, 0), (0, 0), (padding, padding)), mode='constant', constant_values=0)
        out = np.zeros((B, C_out, W_out))
        for i in range(W_out): # a convolution is simply a matrix product of each segment of size (B, C_in, K) in the input and the kernels
            out[:, :, i] = np.einsum('bck,ckx->bx', pad[:, :, i:i+K], kernels.data) # sum over c_in and k
        
        out = Tensor(out, (self, kernels), "conv1d")
        
        def _backward():
            dpad = np.zeros_like(pad)
            dkernels = np.zeros_like(kernels.data)
            for i in range(W_out):
                # the gradient of each slice of the input is simply the product of the kernel and the output gradient (chain rule)
                dpad[:, :, i:i+K] += np.einsum('ckx,bx->bck', kernels.data, out.grad[:, :, i])
                # the gradient of the kernel is the sum of prodcuts of each input slice with the output gradient (chain rule)
                dkernels += np.einsum('bck,bx->ckx', pad[:, :, i:i+K], out.grad[:, :, i])
        
            self.grad += dpad[:, :, padding:W_in+padding] # remove padding
            kernels.grad += dkernels

        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def conv2d(self, kernels:Tensor, padding:tuple[int,int]=(0,0))->Tensor:
        assert self.data.ndim == 4, f"Input tensor must be batched 3d i.e. 4d but got {self.data.ndim=}"
        assert kernels.data.ndim == 4, f"Expected (c_in, k1, k2, c_out) but got {kernels.shape=}"
        assert self.shape[1] == kernels.shape[0], f"Input channel mismatch"
        
        B, C_in, H_in, W_in = self.shape
        C_in, K1, K2, C_out = kernels.shape
        H_out = H_in - K1 + 1 + 2*padding[0]
        W_out = W_in - K2 + 1 + 2*padding[1]
        
        pad = np.pad(self.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant', constant_values=0)
        out = np.zeros((B, C_out, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                out[:, :, i, j] = np.einsum('bcij,cijx->xij', pad[:, :, i:i+K1, j:j+K2], kernels.data)
        
        out = Tensor(out, (self, kernels), "conv2d")
        
        def _backward():
            pass
        
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def __neg__(self)->Tensor:
        out = Tensor(-self.data, (self,), "-")
        
        def _backward():
            self.grad += -out.grad
            
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
    
    def log(self)->Tensor:
        out = Tensor(np.log(self.data), (self,), "log")
        
        def _backward():
            self.grad += out.grad / self.data
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def sum(self, axis:int=None)->Tensor:
        out = Tensor(self.data.sum(axis), (self,), "sum")
        
        def _backward():
            self.grad += broadcast_to(out.grad, self.shape)
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def std(self, axis:int=-1, unbiased:bool=False)->Tensor:
        N = self.data.size if axis is None else self.data.shape[axis]
        mean = self.data.mean(axis, keepdims=True)
        std_dev = np.sqrt(((self.data - mean)**2).sum(axis, keepdims=True) / (N - unbiased))
        out = Tensor(std_dev.squeeze(), (self,), "std")
        
        def _backward():
            self.grad += broadcast_to(out.grad, self.shape) * (self.data - mean) / ((N - unbiased) * std_dev)
            
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
    
    # same with softmax but it would introduce numerical instability 
    def softmax(self)->Tensor:
        shifted_exp = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True)) # numerical stability
        out = Tensor(shifted_exp / shifted_exp.sum(axis=-1, keepdims=True), (self,), "softmax")
        
        def _backward():            
            # ij,ik->ijk computes the outer product of each pair in softmax_output
            jacobian_matrix = np.einsum('ij,ik->ijk', out.data, out.data)
            diag_indices = np.arange(out.shape[-1])
            # subtract softmax output from the diagonal elements
            jacobian_matrix[:, diag_indices, diag_indices] -= out.data
            # Matmul the Jacobian matrix with the gradient of the output
            self.grad += np.einsum('ijk,ik->ij', jacobian_matrix, out.grad)
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    # can be replaced with .softmax().log() but this is more efficient and numerically stable
    def log_softmax(self)->Tensor:
        shifted = self.data - np.max(self.data, axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        out = Tensor(log_probs, (self,), "log_softmax")
        
        def _backward():
            self.grad += out.grad - np.exp(log_probs) * out.grad.sum(axis=-1, keepdims=True)
            
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
    def __truediv__(self, other:int|float|Tensor)->Tensor: return self * other**-1
    
    def mean(self, axis:int=None)->Tensor:
        return self.sum(axis) / (self.data.size if axis is None else self.data.shape[axis])
    
    def __repr__(self):
        data_repr = self.data.__repr__().removeprefix("array")[1:-1] # drop array and parentheses
        grad_repr = ", grad_fn=" + self._grad_fn if self._grad_fn else "" # if grad_fn is empty, don't show it
        return f"Tensor({data_repr}{grad_repr})"
        
    @property
    def shape(self)->tuple[int]:
        return self.data.shape
