import librosa
import numpy as np
import numpy.typing as npt
import scipy.signal

class STFT():

    NAME        = 'Short-Time Fourier Transform'
    UNIT        = 'A'

    frame_size  = 2048
    hop_size    = 1024

    def __init__(self, frame_size: int = 2048, hop_size: int = 1024) -> None:
        self.frame_size = frame_size
        self.hop_size = hop_size

    def compute(self, y: npt.NDArray, sr: int) -> tuple:
        f, t, Y = scipy.signal.stft(
            x=y,
            fs=sr,
            nperseg=self.frame_size, 
            noverlap=self.frame_size - self.hop_size,
            return_onesided=True,
            padded=False
        )
        Y = librosa.stft(y, n_fft=self.frame_size, hop_length=self.hop_size)
        return (Y, t, f)

    def __str__(self) -> str:
        return self.NAME


