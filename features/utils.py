import numpy as np
import librosa
import timeit

def frame(y, frame_size, hop_size):
    xw = np.lib.stride_tricks.sliding_window_view(y, frame_size, axis=-1)
    slices = [slice(None)] * y.ndim
    slices[-1] = slice(0, None, hop_size)
    return xw[tuple(slices)]