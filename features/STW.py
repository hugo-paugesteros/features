import numpy as np
import numpy.typing as npt

from features.PipelineElement import PipelineElement

from . import utils
from . import Waveform
from . import Feature

class STW(Feature):

    y:  npt.NDArray
    sr: int

    FRAME_SIZE: int  = 2048
    HOP_SIZE: int    = 1024

    def __init__(self, frame_size: int, hop_size: int):
        super().__init__()
        
        self.FRAME_SIZE = frame_size
        self.HOP_SIZE   = hop_size

    def compute(self, waveform: Waveform):
        self.y  = utils.frame(waveform.y, frame_size=self.FRAME_SIZE, hop_size=self.HOP_SIZE).swapaxes(-2, -1)
        self.sr = waveform.sr
        self.t  = np.arange(self.y.shape[-1]) * self.HOP_SIZE / waveform.sr

        self.feature = self.y

        return (self.t, self.y)