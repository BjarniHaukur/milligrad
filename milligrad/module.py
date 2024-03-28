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


    