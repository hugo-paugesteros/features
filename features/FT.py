import numpy as np
import numpy.typing as npt

from . import Waveform

class FT():

    def __init__(self):
        pass

    def compute(self, waveform: Waveform):
        padded = np.nan_to_num(waveform.y)
        self.Y = np.fft.rfft(padded * np.hanning(waveform.y.shape[-1]), )
        self.f = np.fft.rfftfreq(waveform.y.shape[-1], 1 / waveform.sr)
        return (self.f, self.Y)