import numpy.typing as npt

class Waveform():

    y:  npt.NDArray
    sr: int

    def __init__(self, y: npt.NDArray, sr: int):
        super().__init__()
        
        self.y  = y
        self.sr = sr