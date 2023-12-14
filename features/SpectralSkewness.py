import numpy as np
import numpy.typing as npt

from .SpectralSpread import SpectralSpread

from .STFT import STFT
from .SpectralCentroid import SpectralCentroid

class SpectralSkewness():

    NAME        = 'Spectral Skewness'
    UNIT        = ''

    frame_size  = 2048
    hop_size    = 1024

    def __init__(self, frame_size: int = 2048, hop_size: int = 1024) -> None:
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.stft = STFT(frame_size=self.frame_size)
        self.SC = SpectralCentroid(mean=False)
        self.SS = SpectralSpread(mean=False)

    def compute(self, y: npt.NDArray, sr: int) -> npt.NDArray:
        Y, _, freq = self.stft.compute(y, sr)
        S = np.abs(Y)
        S /= S.sum(axis=-2, keepdims=True)
        SC = self.SC.compute(y, sr)
        SS = self.SS.compute(y, sr)
        SSk = np.sum(np.subtract.outer(SC, freq).swapaxes(-2, -1)**3 * S, axis=-2) / SS**3
        return np.mean(SSk, axis=-1)

    def __str__(self) -> str:
        return self.NAME


