import unittest
import librosa
import numpy as np
import scipy.signal

from features import STFT

class STFTTest(unittest.TestCase):

    def test(self):
        sr = 44100
        f = 210
        t = np.linspace(0,1,44100)
        y = np.sin(2*np.pi*f*t)

        stft = STFT()
        (Y, t, f) = stft.compute(y, sr)
        Y_true = librosa.stft(y, n_fft=2048, hop_length=1024, center=False)
        scale = np.sqrt(1.0 / scipy.signal.get_window('hann', 2048).sum()**2)
        print(scipy.signal.get_window('hann', 2048))
        print(scipy.signal.stft(y, sr, nperseg=2048, noverlap=1024, padded=False, boundary=None)[2] / scale)
        print(Y_true)
        # self.assertIsNone(np.testing.assert_allclose(Y, Y_true))

if __name__ == "__main__":
    unittest.main()


