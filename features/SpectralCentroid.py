import numpy as np
import numpy.typing as npt
import typing

from . import Feature
from . import STFT, FT

class SpectralCentroid(Feature):

    NAME        = 'Spectral Centroid'
    UNIT        = 'Hz'

    def __init__(self) -> None:
        pass

    @typing.overload
    def compute(self, spectrogram: STFT) -> npt.NDArray:
        ...
    @typing.overload
    def compute(self, spectrogram: FT) -> npt.NDArray:
        ...
    def compute(self, spectrogram: STFT | FT) -> npt.NDArray:
        S = np.abs(spectrogram.Y)

        if(isinstance(spectrogram, FT)):
            self.SC = np.nansum(spectrogram.f * S, axis=-1) / np.nansum(S, axis=-1)
        else:
            # spectrogram.f.shape = (frame_size)
            # S.shape = (frame_size, frames)
            self.SC = np.nansum(spectrogram.f.reshape(-1, 1) * S, axis=-2) / np.nansum(S, axis=-2)
        return self.SC

