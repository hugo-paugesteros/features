import numpy as np
import numpy.typing as npt

class Waveform():

    y:  npt.NDArray
    sr: int
    t:  npt.NDArray

    def __init__(self, y: npt.NDArray, sr: int):
        super().__init__()
        
        self.y  = y
        self.sr = sr
        self.t  = np.arange(y.shape[-1]) / sr

    def compute(self):
        return self