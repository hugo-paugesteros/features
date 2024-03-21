import unittest
import librosa
import numpy as np

from features.RMS import RMS
from features.Waveform import Waveform
from features.STW import STW
from features.STFT import STFT

class PipelineTest(unittest.TestCase):

    def test_sine(self):
        A = 1
        f = 5
        sr = 2048
        t = np.linspace(0,1,sr)
        y = A * np.sin(2*np.pi*f*t)
        waveform = Waveform(y=y, sr=sr)

        pipelines = [
            # [RMS()],
            # [STW(512, 256), RMS()],
            # [STW(512, 256)],
            # [RMS()],
            # [RMS()],
            # [STFT(2048, 1024)],
        ]

        # features = waveform.extract(pipelines)
        # print('----')
        # print(waveform.children)
        # print(features)a

if __name__ == "__main__":
    unittest.main()