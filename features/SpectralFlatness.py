import numpy as np
import numpy.typing as npt

from . import Feature, STFT, FT

class SpectralFlatness(Feature):

    NAME        = 'Spectral Flatness'
    UNIT        = 'Hz'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute(self, spectrogram: STFT | FT) -> npt.NDArray:
        S = np.abs(spectrogram.Y)
        self.SF = np.exp(np.mean(np.log(S), axis=-2)) / np.mean(S, axis=-2)
        return self.SF