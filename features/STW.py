import numpy as np
import numpy.typing as npt

from . import utils
from . import Waveform

class STW():

    y:  npt.NDArray
    sr: int

    frame_size: int  = 2048
    hop_size: int    = 1024

    def __init__(self, frame_size: int, hop_size: int):
        super().__init__()
        
        self.frame_size = frame_size
        self.hop_size   = hop_size

    def compute(self, waveform: Waveform):
        self.y  = utils.frame(waveform.y, frame_size=self.frame_size, hop_size=self.hop_size).swapaxes(-2, -1)
        self.sr = waveform.sr
        self.t  = np.arange(self.y.shape[-1]) * self.hop_size / waveform.sr
        return (self.t, self.y)