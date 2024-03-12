import numpy as np
import numpy.typing as npt

from . import Feature, STFT

class SpectralRolloff(Feature):

    NAME        = 'Spectral Rolloff'
    UNIT        = 'Hz'

    def __init__(self, frame_size: int = 2048, hop_size: int = 1024, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute(self, stft: STFT) -> npt.NDArray:
        S = np.abs(stft.Y)
        cumsum = np.cumsum(S, axis=-2)
        threshold = 0.95 * cumsum[..., -1, :]
        threshold = np.expand_dims(threshold, axis=-2)
        k = np.argmax(cumsum > threshold, axis=-2)
        self.SR = stft.f[k]
        return self.SR