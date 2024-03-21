import numpy as np
import numpy.typing as npt
import typing

from . import Feature, Waveform, STW

class RMS(Feature):

    # NAME        = 'RMS Energy'
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
        axis = -1 if isinstance(audio, Waveform) else -2
        self.rms = 10 * np.log10(np.sqrt(np.nanmean(np.power(audio.y, 2), axis=axis)))
        self.feature = self.rms
        return self.rms