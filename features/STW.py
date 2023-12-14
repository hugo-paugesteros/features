import numpy.typing as npt

from . import utils

class STW():

    y:  npt.NDArray
    sr: int

    def __init__(self, y: npt.NDArray, sr: int, frame_size: int, hop_size: int):
        super().__init__()
        
        self.y  = utils.frame(y, frame_size=frame_size, hop_size=hop_size)
        self.sr = sr