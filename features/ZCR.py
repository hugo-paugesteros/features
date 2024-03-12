import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import librosa
import typing

from . import Feature
from . import Feature, Waveform, STW

class ZCR(Feature):

    NAME        = 'Zero-Crossing Rate'
    UNIT        = 's^-1'

    def __init__(self) -> None:
        super().__init__()

    @typing.overload
    def compute(self, audio: Waveform) -> npt.NDArray:
        ...
    @typing.overload
    def compute(self, audio: STW) -> npt.NDArray:
        ...
    def compute(self, audio: Waveform | STW) -> npt.NDArray:
        axis = -1 if isinstance(audio, Waveform) else -2
        self.zcr = np.sum(np.abs(np.diff(audio.y >= 0, axis=axis)), axis=axis)
        return self.zcr
