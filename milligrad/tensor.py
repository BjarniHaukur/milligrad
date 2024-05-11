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
        return grad.sum(axis=axis).reshape(shape) # broadcast unary dimensions
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
        self.grad = np.zeros_like(data, dtype=np.float32)
        
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
    
    def __truediv__(self, other:Tensor|int|float)->Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), "/")
        
        def _backward():
            self.grad += broadcast_to(out.grad / other.data, self.shape)
            other.grad += broadcast_to(-out.grad * self.data / other.data**2.0, other.shape)
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    # the @ operator is matrix multiplication
    def __matmul__(self, other:Tensor)->Tensor:
        a, b = self.data, other.data
        
        a_vec, b_vec = a.ndim == 1, b.ndim == 1
        if a_vec: a = a[None] # treat a vector as a row matrix
        if b_vec: b = b[..., None] # treat b vector as a column matrix
        
        # add unary dimensions so that a and b have the same dimensionality, summed away later
        a_diff, b_diff = tuple(range(b.ndim - a.ndim)), tuple(range(a.ndim - b.ndim))
        a = a.reshape((1,) * len(a_diff) + a.shape)
        b = b.reshape((1,) * len(b_diff) + b.shape)
        
        c = a @ b # we use the shape of this later to figure out the gradient
        out = Tensor(c.squeeze() if a_vec or b_vec else c, (self, other), "@")
        
        def _backward():
            dc = out.grad.reshape(c.shape) # adds back the unary dimensions if present
            self.grad += (dc @ np.moveaxis(b, -1, -2)).sum(a_diff).sum((0,) if a_vec else ())
            other.grad += (np.moveaxis(a, -1, -2) @ dc).sum(b_diff).sum((-1,) if b_vec else ())
        
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    def conv1d(self, kernels:Tensor, padding:int=0, stride:int=1)->Tensor:
        B, C_in, W_in = self.shape  # Batch size, number of input channels, input width
        _, K, C_out = kernels.shape  # Number of input channels (again), kernel size, number of output channels

        W_out = (W_in - K + 2 * padding) // stride + 1

        pad = np.pad(self.data, ((0, 0), (0, 0), (padding, padding)), mode='constant', constant_values=0)  # i.e. padding the third dimension with 'padding' amount of zeroes on both sides

        Bs, Cs, Ws = pad.strides  # Unpack the strides of the padded input, which define how to navigate the memory layout of the array
        strided = np.lib.stride_tricks.as_strided(pad,
            shape=(B, C_in, W_out, K),  # Create a 4D view of the padded input, where each slice along the third dimension is a segment of length K, effectively "unfolding" the kernel's spatial extent
            strides=(Bs, Cs, Ws * stride, Ws)  # Configure the strides to slide the kernel window over the input, moving K elements at a time along the spatial dimension
        )  

        out = Tensor(np.einsum("biwk,iko->bow", strided, kernels.data, optimize=True), (self, kernels), "conv1d")

        def _backward():
            kernels.grad += np.einsum("biwk,bow->iko", strided, out.grad, optimize=True)  # Convolution operation to update kernels' gradients

            flipped_kernels = np.flip(kernels.data, axis=1)  # Flip the kernels along the spatial dimension for cross-correlation in the backward pass
            padding_out = max((stride * (W_in - 1) + K - W_out) // 2, 0)
            padded_grad = np.pad(out.grad, ((0, 0), (0, 0), (padding_out, padding_out)), mode='constant', constant_values=0)  # Pad by K-1 to mimic how, due to initial padding, edge output pixels can be influenced by just a corner of the kernel

            Bs, Cs, Ws = padded_grad.strides  # Unpack the strides of the padded input, which define how to navigate the memory layout of the array
            strided_grad = np.lib.stride_tricks.as_strided(padded_grad,
                shape=(B, C_out, W_in + 2 * padding, K),  # Create another 4D view, but this time of the padded gradients, to "unroll" the kernel's influence over the input.
                strides=(Bs, Cs, Ws, Ws)  # Configure the strides to slide the kernel window over the gradients, moving K elements at a time along the spatial dimension.
            )

            input_grad = np.einsum('bowk,iko->biw', strided_grad, flipped_kernels, optimize=True)  # Convolution to compute input gradients
            self.grad += input_grad[..., padding:W_in + padding]  # Adjust for the original padding

        if not Tensor._no_grad: self._backward = _backward
        return out
        
    def conv2d(self, kernels:Tensor, padding:tuple[int,int]=(0,0))->Tensor:
        p1, p2 = padding
        B, C_in, H_in, W_in = self.shape  # Batch size, number of input channels, input height, input width
        _, K_H, K_W, C_out = kernels.shape  # Number of input channels (again), kernel height, kernel width, number of output channels

        H_out = H_in - K_H + 1 + 2 * p1
        W_out = W_in - K_W + 1 + 2 * p2

        pad = np.pad(self.data, ((0, 0), (0, 0), (p1, p1), (p2, p2)), mode='constant', constant_values=0)  # i.e. padding the third and fourth dimensions with 'padding' amount of zeroes on both sides

        Bs, Cs, Hs, Ws = pad.strides  # Unpack the strides of the padded input, which define how to navigate the memory layout of the array
        strided = np.lib.stride_tricks.as_strided(pad,
            shape=(B, C_in, H_out, W_out, K_H, K_W),  # Create a 6D view of the padded input, where each slice along the third and fourth dimensions is a segment of size K_H x K_W, effectively "unfolding" the kernel's spatial extent
            strides=(Bs, Cs, Hs, Ws, Hs, Ws),  # Configure the strides to slide the kernel window over the input, moving K_H x K_W elements at a time along the spatial dimensions
        )

        out = Tensor(np.einsum('bchwij,cijk->bkhw', strided, kernels.data, optimize=True), (self, kernels), "conv2d")

        def _backward():
            kernels.grad += np.einsum('bchwij,bkhw->cijk', strided, out.grad, optimize=True)  # Convolution operation to update kernels' gradients

            flipped_kernels = np.flip(kernels.data, axis=(1, 2))  # Flip the kernels along the spatial dimensions for cross-correlation in the backward pass
            padded_out_grad = np.pad(out.grad, ((0, 0), (0, 0), (K_H - 1, K_H - 1), (K_W - 1, K_W - 1)), mode='constant', constant_values=0)  # Pad by K_H-1 and K_W-1 to mimic how, due to initial padding, edge output pixels can be influenced by just a corner of the kernel

            Bs, Cs, Hs, Ws = padded_out_grad.strides  # Unpack the strides of the padded input, which define how to navigate the memory layout of the array
            strided_grad = np.lib.stride_tricks.as_strided(padded_out_grad,
                shape=(B, C_out, H_in + 2 * p1, W_in + 2 * p2, K_H, K_W),  # Create another 6D view, but this time of the padded gradients, to "unroll" the kernel's influence over the input.
                strides=(Bs, Cs, Hs, Ws, Hs, Ws),  # Configure the strides to slide the kernel window over the gradients, moving K_H x K_W elements at a time along the spatial dimensions.
            )

            dpad = np.einsum('bkhwij,cijk->bchw', strided_grad, flipped_kernels, optimize=True)  # Convolution to compute input gradients
            self.grad += dpad[:, :, p1:H_in + p1, p2:W_in + p2]  # Adjust for the original padding

        if not Tensor._no_grad: self._backward = _backward
        return out

    def maxpool1d(self, kernel_size:int=2)->Tensor:
        assert self.data.ndim == 3, f"Input tensor must be batched 2d i.e. 3d but got {self.data.ndim=}"
        assert self.shape[-1] % kernel_size == 0 and kernel_size > 0, "The length of the sequence must be divisible by the kernel size" 
        
        B, C_in, W_in = self.shape
        W_out = W_in // kernel_size

        Bs, Cs, Ws = self.data.strides  # Unpack the strides of the padded input, which define how to navigate the memory layout of the array
        strided = np.lib.stride_tricks.as_strided(self.data,
            shape=(B, C_in, W_out, kernel_size),  # Generate a new view that groups input elements into blocks of 'kernel_size' along the width
            strides=(Bs, Cs, Ws * kernel_size, Ws)  # Adjust strides to step over 'kernel_size' elements along the width for each sliding window position
        )  
        
        max_indices = strided.argmax(axis=-1, keepdims=True) # for putting along axis later
        out = Tensor(strided.max(axis=-1), (self,), "maxpool1d")
        
        def _backward():
            strided_grad = np.zeros_like(strided)
            np.put_along_axis(strided_grad, max_indices, out.grad[..., None], axis=-1)  # Propagate gradients to positions where the max was
            self.grad += strided_grad.reshape(B, C_in, W_out * kernel_size) 

        if not Tensor._no_grad: self._backward = _backward
        return out
        
    def maxpool2d(self, kernel_size:tuple[int,int]=(2,2))->Tensor:
        assert self.data.ndim == 4, f"Input tensor must be batched 3d i.e. 4d but got {self.data.ndim=}"
        assert self.shape[-2] % kernel_size[0] == 0 and self.shape[-1] % kernel_size[1] == 0, "The width and height of the image must be divisible by the kernel dimensions"
        
        B, C, H_in, W_in = self.shape
        K_H, K_W = kernel_size
        H_out, W_out = H_in // K_H, W_in // K_W
        
        Bs, Cs, Hs, Ws = self.data.strides  # Unpack the strides of the padded input, which define how to navigate the memory layout of the array
        strided = np.lib.stride_tricks.as_strided(self.data,
            shape=(B, C, H_out, W_out, K_H, K_W),  # Generate a new view that groups input elements into blocks of (K_H, K_W) along the height and width
            strides=(Bs, Cs, Hs * K_H, Ws * K_W, Hs, Ws),  # Adjust strides to step over (K_H, K_W) elements along the height and width for each sliding window position
        ).reshape(B, C, H_out, W_out, K_H * K_W)  # Reshape strided data to a flat (K_H * K_W) dimension per pool area for applying max operation.
        
        max_indices = strided.reshape(B, C, H_out, W_out, K_H * K_W).argmax(axis=-1, keepdims=True)
        out = Tensor(strided.max(axis=-1), (self,), "maxpool2d")
        
        def _backward():
            strided_grad = np.zeros_like(strided)
            np.put_along_axis(strided_grad, max_indices, out.grad[..., None], axis=-1)  # Propagate gradients to positions where the max was
            strided_grad = strided_grad.reshape(B, C, H_out, W_out, K_H, K_W)
            self.grad += strided_grad.swapaxes(3,4).reshape(B, C, H_in, W_in)  # Swapaxes is crucial as it effectively reverses the striding operation when followed by the reshape operation
            
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
            self.grad += out.grad.reshape(self.shape)
            
        if not Tensor._no_grad: self._backward = _backward
        return out
    
    ###################################################################################
    ############## The following functions simply apply other functions ###############
    ###################################################################################
    
    def __sub__(self, other:Tensor|int|float)->Tensor: return self + (-other)
    def __radd__(self, other:int|float)->Tensor: return self + other
    def __rsub__(self, other:int|float)->Tensor: return other + (-self)
    def __rmul__(self, other:int|float)->Tensor: return self * other
    
    def mean(self, axis:int=None)->Tensor:
        return self.sum(axis) / (self.data.size if axis is None else self.data.shape[axis])
    
    def __repr__(self):
        data_repr = self.data.__repr__().removeprefix("array")[1:-1] # drop array and parentheses
        grad_repr = ", grad_fn=" + self._grad_fn if self._grad_fn else "" # if grad_fn is empty, don't show it
        return f"Tensor({data_repr}{grad_repr})"
        
    @property
    def shape(self)->tuple[int]:
        return self.data.shape
