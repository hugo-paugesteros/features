import numpy as np
import numpy.typing as npt

from .STFT import STFT

class SpectralCentroid():

    NAME        = 'Spectral Centroid'
    UNIT        = 'Hz'

    frame_size  = 2048
    hop_size    = 1024
    mean        = True

    def __init__(self, frame_size: int = 2048, hop_size: int = 1024, mean=True) -> None:
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.mean = mean
        self.stft = STFT(frame_size=self.frame_size, hop_size=self.hop_size)

    def compute(self, y: npt.NDArray, sr: int) -> npt.NDArray:
        Y, _, freq = self.stft.compute(y, sr)
        S = np.abs(Y)
        SC = np.sum(freq.reshape(-1, 1) * S, axis=-2) / np.sum(S, axis=-2)
        if self.mean:
            return np.mean(SC, axis=-1)
        return SC

    def __str__(self) -> str:
        return self.NAME

