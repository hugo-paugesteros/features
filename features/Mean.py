import numpy as np
import numpy.typing as npt

from . import Waveform
from . import RMS
from . import ZCR
from . import SpectralCentroid
from . import SpectralSpread
from . import SpectralSkewness
from . import SpectralKurtosis
from . import SpectralSlope
from . import SpectralDecrease
from . import SpectralRolloff
from . import SpectralVariation
from . import SpectralFlux
from . import SpectralFlatness
from . import SpectralCrest

class Mean():

    def __init__(self):
        pass

    def compute(self, feature) -> None:
        if isinstance(feature, RMS):
            return np.nanmean(feature.rms, axis=-1)
        if isinstance(feature, ZCR):
            return np.nanmean(feature.zcr, axis=-1)
        if isinstance(feature, SpectralCentroid):
            return np.nanmean(feature.SC, axis=-1)
        if isinstance(feature, SpectralSpread):
            return np.nanmean(feature.SS, axis=-1)
        if isinstance(feature, SpectralSkewness):
            return np.nanmean(feature.SSk, axis=-1)
        if isinstance(feature, SpectralKurtosis):
            return np.nanmean(feature.SK, axis=-1)
        if isinstance(feature, SpectralSlope):
            return np.nanmean(feature.SS, axis=-1)
        if isinstance(feature, SpectralDecrease):
            return np.nanmean(feature.SD, axis=-1)
        if isinstance(feature, SpectralRolloff):
            return np.nanmean(feature.SR, axis=-1)
        if isinstance(feature, SpectralVariation):
            return np.nanmean(feature.SV, axis=-1)
        if isinstance(feature, SpectralFlux):
            return np.nanmean(feature.SF, axis=-1)
        if isinstance(feature, SpectralFlatness):
            return np.nanmean(feature.SF, axis=-1)
        if isinstance(feature, SpectralCrest):
            return np.nanmean(feature.SC, axis=-1)