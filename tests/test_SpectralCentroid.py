import unittest
import librosa
import numpy as np

from features.SpectralCentroid import SpectralCentroid
from features.FT import FT
from features.STFT import STFT
from features.Waveform import Waveform

class SpectralCentroidTest(unittest.TestCase):

    def test_sine(self):
        sr = 44100
        f = 210
        t = np.linspace(0,1,44100)
        y = np.sin(2*np.pi*f*t)

        spectrogram = FT(Waveform(y, sr))

        SC = SpectralCentroid().compute(spectrogram)
        SC_true = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=len(y), hop_length=len(y), center=False)[0]
        
        self.assertIsNone(np.testing.assert_allclose(SC, SC_true, atol=1e-3))

    def test_framed_sine(self):
        sr = 44100
        f = 210
        t = np.linspace(0,1,44100)
        y = np.sin(2*np.pi*f*t)

        spectrogram = STFT(Waveform(y, sr), 2048, 1024)

        SC = SpectralCentroid().compute(spectrogram)
        SC_true = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=1024)[0]
        
        self.assertIsNone(np.testing.assert_allclose(SC, SC_true))

if __name__ == "__main__":
    unittest.main()
