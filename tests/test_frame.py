import unittest
import numpy as np
import librosa
import timeit

from features.utils import frame

class RMSTest(unittest.TestCase):

    def test_equality(self):
        y = np.arange(50000)
        frame_size  = 2048
        hop_size    = 1024

        result = frame(y, frame_size=frame_size, hop_size=hop_size)
        expected_result = librosa.util.frame(y,frame_length=frame_size, hop_length=hop_size)
        self.assertIsNone(np.testing.assert_allclose(result, expected_result, atol=1e-2))

    def test_time(self):
        print(
            timeit.timeit(
                'frame(y, 2048, 1024)',
                'y = np.random.rand(500000)',
                number=10000,
                globals=globals()
            )
        )

        print(
            timeit.timeit(
                'librosa.util.frame(y, frame_length=2048, hop_length=1024)',
                'y = np.random.rand(500000)',
                number=10000,
                globals=globals()
            )
        ) 

if __name__ == "__main__":
    unittest.main()