from abc import ABC, abstractmethod

from milligrad.tensor import Tensor

import numpy as np

class Optimizer(ABC):
    def __init__(self, params:list[Tensor]):
        self.params = params

    @abstractmethod
    def step(self): 
        pass

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

class GD(Optimizer):
    def __init__(self, params:list[Tensor], lr:float=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
            
class Momentum(Optimizer):
    def __init__(self, params:list[Tensor], lr:float=0.01, momentum:float=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * p.grad
            p.data -= self.velocities[i] 
            
class AdaGrad(Optimizer):
    def __init__(self, params:list[Tensor], lr:float=0.01, epsilon:float=1e-8):
        super().__init__(params)
        self.lr = lr
        self.epsilon = epsilon
        self.cache = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.cache[i] += p.grad ** 2
            p.data -= self.lr * p.grad / (np.sqrt(self.cache[i]) + self.epsilon)
               
class RMSProp(Optimizer):
    def __init__(self, params:list[Tensor], lr:float=0.01, decay_rate:float=0.99, epsilon:float=1e-8):
        super().__init__(params)
        self.lr = lr
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.cache[i] = self.decay_rate * self.cache[i] + (1 - self.decay_rate) * p.grad ** 2
            p.data -= self.lr * p.grad / (np.sqrt(self.cache[i]) + self.epsilon)
            
# Adam is essentially AdaGrad and RMSProp with bias correction
class Adam(Optimizer):
    def __init__(self, params:list[Tensor], lr:float=0.001, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
