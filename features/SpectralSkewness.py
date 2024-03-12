import numpy as np
import numpy.typing as npt

from .SpectralSpread import SpectralSpread

from .STFT import STFT
from .FT import FT
from .SpectralCentroid import SpectralCentroid
from .SpectralSpread import SpectralSpread

class SpectralSkewness():

    NAME        = 'Spectral Skewness'
    UNIT        = ''

    def __init__(self) -> None:
        pass

    def compute(self, spectrogram: STFT | FT) -> npt.NDArray:
        S = np.abs(spectrogram.Y)
        S /= S.sum(axis=-2, keepdims=True)
        SC = SpectralCentroid().compute(spectrogram)
        SS = SpectralSpread().compute(spectrogram)
        self.SSk = np.sum(np.subtract.outer(SC, spectrogram.f).swapaxes(-2, -1)**3 * S, axis=-2) / SS**3
        return self.SSk


