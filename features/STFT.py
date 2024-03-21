import numpy as np
import numpy.typing as npt
import scipy.signal

from . import Waveform, Feature, STW, FT

class STFT(Feature):

    FRAME_SIZE: int  = 2048
    HOP_SIZE: int    = 1024

    def __init__(self, frame_size: int = 2048, hop_size: int = 1024) -> None:
        super().__init__()
        self.FRAME_SIZE = frame_size
        self.HOP_SIZE = hop_size

    def compute(self, waveform: Waveform):
        self.stw = self.PARENT.add_child(STW(self.FRAME_SIZE, self.HOP_SIZE))
        self.FT = self.stw.add_child(FT())

        self.f, self.t, self.Y = scipy.signal.stft(
            x=waveform.y,
            fs=waveform.sr,
            nperseg=self.FRAME_SIZE, 
            noverlap=self.FRAME_SIZE - self.HOP_SIZE,
            return_onesided=True,
            padded=False
        )
        self.feature = self.Y
        return (self.f, self.t, self.Y)

