import numpy as np
import numpy.typing as npt

from . import Feature, STFT

class SpectralDecrease(Feature):

    NAME        = 'Spectral Decrease'
    UNIT        = ''

    def __init__(self, frame_size: int = 2048, hop_size: int = 1024, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute(self, stft: STFT) -> npt.NDArray:
        S = np.abs(stft.Y)
        k = np.arange(S.shape[-2]).reshape(-1, 1)
        # Do not divide by zero, please
        k[0,0] = 1
        self.SD = np.sum( (S - S[..., 0:1, :]) / k, axis=-2) / np.sum(S, axis=-2)
        return self.SD