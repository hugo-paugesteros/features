import numpy as np
import numpy.typing as npt

from . import Feature, STFT, FT

class SpectralFlux(Feature):

    NAME        = 'Spectral Flux'
    UNIT        = 'Hz'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute(self, spectrogram: STFT | FT) -> npt.NDArray:
        S = np.abs(spectrogram.Y)
        self.SF = np.sqrt(np.sum((S[..., 1:] - S[..., :-1])**2, axis=-2))
        return self.SF