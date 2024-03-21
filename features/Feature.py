import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from features.PipelineElement import PipelineElement

from . import Waveform

class Feature(PipelineElement):

    # NAME = ''
    feature = None

    def __init__(self):
        super().__init__()
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

    # def __str__(self) -> str:
    #     return self.NAME