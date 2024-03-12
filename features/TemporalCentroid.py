import numpy as np
import numpy.typing as npt

from . import Waveform

class TemporalCentroid():

    def __init__(self):
        super().__init__()

    def compute(self, waveform: Waveform):
        self.temporal_centroid = np.nansum(waveform.t * waveform.y, axis=-1) / np.nansum(waveform.y, axis=-1)
        return self.temporal_centroid