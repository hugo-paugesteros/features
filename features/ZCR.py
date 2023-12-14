import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import librosa

from . import Feature

class ZCR(Feature):

    NAME        = 'Zero-Crossing Rate'
    UNIT        = 's^-1'

    frame_size  = 2048
    hop_size    = 1024

    def __init__(self, frame_size: int = 2048, hop_size: int = 1024) -> None:
        super().__init__()
        self.frame_size = frame_size
        self.hop_size = hop_size

    def _compute(self, y: npt.NDArray, sr: int) -> npt.NDArray:
        self.y = y
        self.frames = librosa.util.frame(
            y, 
            frame_length=self.frame_size, 
            hop_length=self.hop_size
        )
        # self.zcr = np.abs(np.diff(np.sign(self.frames), axis=0)) / 2
        # self.zcr = np.diff(self.frames >= 0, axis=0)
        self.zcr = np.diff(np.sign(self.frames), axis=-2) != 0
        self.zcr = np.diff(self.sign(self.frames), axis=-2) != 0
        # self.zcr = ((self.frames[:-1, :] * self.frames[1:, :]) < 0)
        # * 2048 / self.frame_size
        return np.sum(self.zcr, axis=-2)

    def sign(self, y):
        sign = np.sign(y)
        sign[sign == 0] = 1
        return sign


def zcr(window):
    window += 0.001
    # return np.diff(window >= 0, append=[False])
    return np.sum(np.abs(np.diff(np.sign(window)))) / 2
    return np.diff(np.sign(window), append=[0])
    return np.sum(np.abs(np.diff(np.sign(window), append=[0]))/2)
    # return np.sum(np.diff(window >= 0))
    # return np.sum(np.abs(np.diff(np.sign(window))))/2
