import numpy as np
import numpy.typing as npt
import scipy.signal

from . import Waveform

class STFT():

    frame_size: int  = 2048
    hop_size: int    = 1024

    def __init__(self, frame_size: int = 2048, hop_size: int = 1024) -> None:
        self.frame_size = frame_size
        self.hop_size = hop_size

    def compute(self, waveform: Waveform):
        self.f, self.t, self.Y = scipy.signal.stft(
            x=waveform.y,
            fs=waveform.sr,
            nperseg=self.frame_size, 
            noverlap=self.frame_size - self.hop_size,
            return_onesided=True,
            padded=False
        )
        return (self.f, self.t, self.Y)

