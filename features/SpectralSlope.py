import numpy as np
import numpy.typing as npt
import librosa

from . import Feature, STFT

class SpectralSlope(Feature):

    NAME        = 'Spectral Slope'
    UNIT        = 'Hz^-1'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute(self, stft: STFT) -> npt.NDArray:
        S = np.abs(stft.Y)
        f_mean = np.mean(stft.f)
        self.SS = np.sum((stft.f - f_mean).reshape(-1, 1) * (S - np.mean(S, axis=-2, keepdims=True)), axis=-2) / np.sum((stft.f - f_mean)**2)
        return self.SS