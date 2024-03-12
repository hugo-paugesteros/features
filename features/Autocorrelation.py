import numpy as np
import numpy.typing as npt
import typing

from . import Feature, Waveform, STW

class Autocorrelation(Feature):

    NAME        = 'Autocorrelation'
    UNIT        = ''

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @typing.overload
    def compute(self, audio: Waveform) -> npt.NDArray:
        ...
    @typing.overload
    def compute(self, audio: STW) -> npt.NDArray:
        ...
    def compute(self, audio: Waveform | STW) -> npt.NDArray:
        if isinstance(audio, Waveform):
            autocorrelation = np.correlate(audio.y, audio.y, 'same')[:12]
        else:
            autocorrelation = np.zeros((12, audio.y.shape[1]))
            for i in range(audio.y.shape[1]):
                autocorrelation[:, i] = np.correlate(audio.y[:, i], audio.y[:, i], 'same')[:12]
        return autocorrelation