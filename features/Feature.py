import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from . import Waveform

class Feature(ABC):

    NAME = ''
    mean = True

    def __init__(self, mean: bool = True):
        self.mean = mean
        pass

    # @abstractmethod
    # def _compute(self, y: npt.NDArray) -> npt.NDArray:
    #     pass

    # @abstractmethod
    # def _compute(self, waveform: Waveform) -> npt.NDArray:
    #     pass
    
    # def compute(self, **kwargs):
    #     if self.mean:
    #         return np.mean(self._compute(**kwargs), axis=-1)
    #     else:
    #         return self._compute(**kwargs)

    def __str__(self) -> str:
        return self.NAME