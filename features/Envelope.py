import numpy as np
import numpy.typing as npt

from . import Waveform

from . import STW, RMS

class Envelope():

    frame_size = 4096
    hop_size = 2048

    def __init__(self):
        super().__init__()

    def compute(self, waveform: Waveform):
        stw = STW(frame_size=self.frame_size, hop_size=self.hop_size)
        stw.compute(waveform)
        
        rms = RMS()
        rms.compute(stw)

        self.t = stw.t
        self.envelope = rms.rms
        return self.envelope

