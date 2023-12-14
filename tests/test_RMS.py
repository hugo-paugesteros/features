import unittest
import librosa
import numpy as np

from features.RMS import RMS
from features.Waveform import Waveform
from features.STW import STW

class RMSTest(unittest.TestCase):

    def test_sine(self):
        A = 1
        f = 5
        sr = 2048
        t = np.linspace(0,1,sr)
        y = A * np.sin(2*np.pi*f*t)
        waveform = Waveform(y=y, sr=sr)

        rms = RMS()
        result = rms.compute(waveform)
        expected_result = 10 * np.log10(A / np.sqrt(2))
        self.assertIsNone(np.testing.assert_allclose(result, expected_result, atol=1e-2))

    def test_framed_sine(self):
        A = 1
        f = 5
        sr = 22050
        t = np.linspace(0,1,sr)
        y = A * np.sin(2*np.pi*f*t)
        waveform = STW(y=y, sr=sr, frame_size=2048, hop_size=1024)

        rms = RMS()
        result = rms.compute(waveform)
        expected_result = 10 * np.log10(librosa.feature.rms(y=y, frame_length=2048, hop_length=1024, center=False)[0])
        self.assertIsNone(np.testing.assert_allclose(result, expected_result, atol=1e-2))

if __name__ == "__main__":
    unittest.main()