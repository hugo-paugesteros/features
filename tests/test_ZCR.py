import unittest
import numpy as np

from src.ZCR import ZCR

class ZCRTest(unittest.TestCase):

    def test_simple(self):
        y = np.array([5,0,-5,0,5,-5])
        zcr = ZCR(frame_size=6, hop_size=2048)
        result = zcr.compute(y)
        self.assertIsNone(np.testing.assert_allclose(result, [3], atol=0))

    def test_start_with_zero(self):
        y = np.array([0,1,-1,-1,0,1])
        zcr = ZCR(frame_size=6, hop_size=2048)
        result = zcr.compute(y)
        self.assertIsNone(np.testing.assert_allclose(result, [2], atol=0))

    def test(self):
        f = 210
        t = np.linspace(0,1,2048)
        y = np.sin(2*np.pi*f*t)

        zcr = ZCR(frame_size=2048, hop_size=2048)
        result = zcr.compute(y)
        self.assertIsNone(np.testing.assert_allclose(result, [2*f], atol=1))

if __name__ == "__main__":
    unittest.main()

