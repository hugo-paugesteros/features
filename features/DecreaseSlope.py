import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

from . import Envelope

class DecreaseSlope():

    def __init__(self):
        super().__init__()

    def compute(self, envelope: Envelope):
        diff = np.diff(envelope.envelope, axis=-1)
        fliped = np.flip(diff, axis=-1)
        t_start = envelope.envelope.shape[-1] - np.argmax(fliped > 0, axis=-1)
        t_end = np.argmax(np.isnan(envelope.envelope), axis=-1)
        
        self.decrease_slope = envelope.t[t_end] - envelope.t[t_start]
        return self.decrease_slope