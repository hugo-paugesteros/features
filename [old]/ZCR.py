import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import librosa
import typing

from . import Feature
from . import Feature, Waveform, STW

class ZCR(Feature):

    NAME        = 'Zero-Crossing Rate'
    UNIT        = 's^-1'

    def __init__(self) -> None:
        super().__init__()

    @typing.overload
    def compute(self, audio: Waveform) -> npt.NDArray:
        ...
    @typing.overload
    def compute(self, audio: STW) -> npt.NDArray:
        ...
    def compute(self, audio: Waveform | STW) -> npt.NDArray:
        axis = -1 if isinstance(audio, Waveform) else -2
        # zcr = np.sum(np.diff(np.sign(audio.y), axis=axis) != 0, axis=axis)
        zcr = np.sum(np.abs(np.diff(audio.y >= 0, axis=axis)))
        return zcr

    # def _compute(self, y: npt.NDArray, sr: int) -> npt.NDArray:
    #     self.y = y
    #     self.frames = librosa.util.frame(
    #         y, 
    #         frame_length=self.frame_size, 
    #         hop_length=self.hop_size
    #     )
    #     # self.zcr = np.abs(np.diff(np.sign(self.frames), axis=0)) / 2
    #     # self.zcr = np.diff(self.frames >= 0, axis=0)
    #     self.zcr = np.diff(np.sign(self.frames), axis=-2) != 0
    #     self.zcr = np.diff(self.sign(self.frames), axis=-2) != 0
    #     # self.zcr = ((self.frames[:-1, :] * self.frames[1:, :]) < 0)
    #     # * 2048 / self.frame_size
    #     return np.sum(self.zcr, axis=-2)

    def sign(self, y):
        sign = np.sign(y)
        sign[sign == 0] = 1
        return sign


# def zcr(window):
#     window += 0.001
#     # return np.diff(window >= 0, append=[False])
#     return np.sum(np.abs(np.diff(np.sign(window)))) / 2
#     return np.diff(np.sign(window), append=[0])
#     return np.sum(np.abs(np.diff(np.sign(window), append=[0]))/2)
#     # return np.sum(np.diff(window >= 0))
#     # return np.sum(np.abs(np.diff(np.sign(window))))/2
