import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

from . import Envelope

class Attack():

    def __init__(self):
        super().__init__()

    def compute(self, envelope: Envelope):
        # maximum = np.nanmax(envelope.envelope, axis=-1)
        t_end = envelope.t[np.argmax(envelope.envelope, axis=-1)]
        # t_start = envelope.t[np.argmin(envelope.envelope > maximum * 0.1, axis=-1)]
        t_start = 0
        self.attack = t_end - t_start
        return self.attack