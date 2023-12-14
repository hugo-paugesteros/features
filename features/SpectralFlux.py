import numpy as np
import numpy.typing as npt

from . import Feature, STFT

class SpectralFlux(Feature):

    NAME        = 'Spectral Flux'
    UNIT        = 'Hz'

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
        SF = np.sqrt(np.sum((S[..., 1:] - S[..., :-1])**2, axis=-2))
        return SF