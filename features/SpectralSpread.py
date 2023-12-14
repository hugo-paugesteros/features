import numpy as np
import numpy.typing as npt

from .STFT import STFT
from .SpectralCentroid import SpectralCentroid

class SpectralSpread():

    NAME        = 'Spectral Centroid'
    UNIT        = 'Hz'

    frame_size  = 2048
    hop_size    = 1024
    mean        = True

    def __init__(self, frame_size: int = 2048, hop_size: int = 1024, mean=True) -> None:
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.stft = STFT(frame_size=self.frame_size)
        self.SC = SpectralCentroid(mean=False)
        self.mean = mean

    def compute(self, y: npt.NDArray, sr: int) -> npt.NDArray:
        Y, _, freq = self.stft.compute(y, sr)
        S = np.abs(Y) 
        SC = self.SC.compute(y, sr)
        SS = np.sqrt(np.sum(np.subtract.outer(SC, freq).swapaxes(-2, -1)**2 * S, axis=-2))
        if self.mean:
            return np.mean(SS, axis=-1)
        return SS

    def __str__(self) -> str:
        return self.NAME


