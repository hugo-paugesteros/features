import numpy as np
import numpy.typing as npt
import librosa

from . import Feature, STFT

class SpectralSlope(Feature):

    NAME        = 'Spectral Slope'
    UNIT        = 'Hz^-1'

    frame_size  = 2048
    hop_size    = 1024

    def __init__(self, frame_size: int = 2048, hop_size: int = 1024, **kwargs) -> None:
        super().__init__(**kwargs)
        self.frame_size = frame_size
        self.hop_size = hop_size

        self.stft = STFT(frame_size=self.frame_size, hop_size=self.hop_size)

    def _compute(self, y: npt.NDArray, sr: int) -> npt.NDArray:
        Y, _, f = self.stft.compute(y, sr)
        S = np.abs(Y)
        f_mean = np.mean(f)
        SS = np.sum((f - f_mean).reshape(-1, 1) * (S - np.mean(S, axis=-2, keepdims=True)), axis=-2) / np.sum((f - f_mean)**2)
        return SS