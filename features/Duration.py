import numpy as np
import numpy.typing as npt

from . import Waveform

class Duration():

    def __init__(self):
        super().__init__()

    def compute(self, waveform: Waveform):
        self.duration = np.argmax(np.isnan(waveform.y), axis=-1) / waveform.sr
        return self.duration