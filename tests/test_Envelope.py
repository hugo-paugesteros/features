import unittest
import numpy as np
import matplotlib.pyplot as plt

from features.Envelope import Envelope

class RMSTest(unittest.TestCase):

    def test_equality(self):
        A = 1
        f = 50
        sr = 1024*30
        t = np.linspace(0,1,sr)
        y = A * np.sin(2*np.pi*f*t) * np.sin(2*np.pi*5*t)
        frame_size  = 512
        hop_size    = frame_size

        result = Envelope(y, sr, frame_size=frame_size, hop_size=hop_size)
        plt.plot(t, np.abs(y))
        plt.step(np.linspace(0,1,int(sr/hop_size)), result.y, where='post')
        print(result.y)
        plt.show()


if __name__ == "__main__":
    unittest.main()