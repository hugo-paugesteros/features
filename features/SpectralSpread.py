import numpy as np
import numpy.typing as npt

from . import Feature
from .STFT import STFT
from .FT import FT
from .SpectralCentroid import SpectralCentroid

class SpectralSpread(Feature):

    NAME        = 'Spectral Spread'
    UNIT        = 'Hz'

    def __init__(self) -> None:
        pass

    def compute(self, spectrogram: STFT | FT) -> npt.NDArray:
        S = np.abs(spectrogram.Y) 
        SC = SpectralCentroid().compute(spectrogram)

        '''
        SC.shape = (frames)
        spectrogram.f.shape = (frame_size)
        (SC - spectrogram.f).shape = (frame_size, frames)
        SS.shape = (frames)
        '''
        if isinstance(spectrogram, FT):
            self.SS = np.sqrt(np.sum(np.subtract.outer(SC, spectrogram.f)**2 * S, axis=-1) / np.sum(S, axis=-1))
        else:
            # SS = np.sqrt(np.sum(np.subtract.outer(spectrogram.f, SC)**2 * S, axis=-2) / np.sum(S))
            self.SS = np.sqrt(np.sum(np.subtract.outer(SC, spectrogram.f).swapaxes(-2, -1)**2 * S, axis=-2) / np.sum(S, axis=-2))
        return self.SS