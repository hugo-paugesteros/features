import unittest
import librosa
import numpy as np

from features import SpectralCentroid

class SpectralCentroidTest(unittest.TestCase):

    def test_sine(self):
        sr = 44100
        f = 210
        t = np.linspace(0,1,44100)
        y = np.sin(2*np.pi*f*t)

        SC = SpectralCentroid(mean=False).compute(y, sr)
        SC_true = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=1024)[0]
        print(SC)
        self.assertIsNone(np.testing.assert_allclose(SC, SC_true))

if __name__ == "__main__":
    unittest.main()
