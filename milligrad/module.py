from milligrad.tensor import Tensor

from abc import ABC, abstractmethod

import numpy as np


class Module(ABC):

    @abstractmethod
    def __call__(self, x:Tensor)->Tensor:
        pass
            
    @abstractmethod
    def parameters(self)->list[Tensor]:
        pass
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    