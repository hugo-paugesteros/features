import unittest
import librosa
import numpy as np

from features.Waveform import Waveform
from features.Envelope import Envelope
from features.Attack import Attack

class RMSTest(unittest.TestCase):

    def test_extract(self):
        y, sr = librosa.load('/home/hugo/Thèse/segmentation/processed/V1F1D1T1.wav', mono=True)
        waveform = Waveform(y=y, sr=sr)
        
        envelope = Envelope()
        envelope.compute(waveform)
        
        attack = Attack()
        attack.compute(envelope)

        print(attack.attack)

if __name__ == "__main__":
    unittest.main()