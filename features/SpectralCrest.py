import numpy as np
import numpy.typing as npt

from . import Feature, STFT, FT

class SpectralCrest(Feature):

    NAME        = 'Spectral Crest'
    UNIT        = 'Hz'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute(self, spectrogram: STFT | FT) -> npt.NDArray:
        S = np.abs(spectrogram.Y)
        self.SC = np.max(S, axis=-2) / np.mean(S, axis=-2)
        return self.SC