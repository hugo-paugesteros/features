import unittest
import numpy as np
import librosa

from features.ZCR import ZCR
from features.Waveform import Waveform
from features.STW import STW

class ZCRTest(unittest.TestCase):

    def test_simple(self):
        y = np.array([5.,0,-5,0,5,-5])
        waveform = Waveform(y=y, sr=3)
        zcr = ZCR()
        result = zcr.compute(waveform)
        self.assertIsNone(np.testing.assert_allclose(result, [3], atol=0))

    def test_start_with_zero(self):
        waveform = Waveform(y=np.array([0,1,-1,-1,0,1]), sr=1)
        zcr = ZCR()
        result = zcr.compute(waveform)
        self.assertIsNone(np.testing.assert_allclose(result, [2], atol=0))

    def test_sine(self):
        f = 210
        t = np.linspace(0,1,2048)
        y = np.sin(2*np.pi*f*t)
        waveform = Waveform(y=y, sr=1)

        zcr = ZCR()
        result = zcr.compute(waveform)
        self.assertIsNone(np.testing.assert_allclose(result, [2*f], atol=0))

    def test_framed_sine(self):
        f = 210
        sr = 22050
        t = np.linspace(0,1,sr)
        y = np.sin(2*np.pi*f*t)
        stw = STW(y=y, sr=sr, frame_size=2048, hop_size=1024)

        zcr = ZCR()
        result = zcr.compute(stw)
        expected = [int(2048/sr * 2*f)] * len(result)
        self.assertIsNone(np.testing.assert_allclose(result, expected, atol=1))

if __name__ == "__main__":
    unittest.main()

