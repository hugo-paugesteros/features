import numpy as np
import librosa
import timeit

def frame(y, frame_size, hop_size):
    return np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size].T