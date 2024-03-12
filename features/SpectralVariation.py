import numpy as np
import numpy.typing as npt

from . import Feature, STFT

class SpectralVariation(Feature):

    NAME        = 'Spectral Variation'
    UNIT        = 'Hz'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute(self, stft: STFT) -> npt.NDArray:
        S = np.abs(stft.Y)
        
        cumsum = np.cumsum(S, axis=-2)
        threshold = 0.95 * cumsum[..., -1, :]
        threshold = np.expand_dims(threshold, axis=-2)
        k = np.argmax(cumsum > threshold, axis=-2)
        self.SV = 1 - np.sum(S[..., 1:] * S[..., :-1], axis=-2) / np.sqrt(np.sum(S[..., :-1]**2, axis=-2) * np.sum(S[..., 1:]**2, axis=-2))
        return self.SV