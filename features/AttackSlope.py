import numpy as np
import numpy.typing as npt

from . import Envelope

class AttackSlope():

    def __init__(self):
        super().__init__()

    def compute(self, envelope: Envelope):
        slices = [slice(None)] * envelope.envelope.ndim
        slices[-1] = 0
        minimum = envelope.envelope[tuple(slices)]
        maximum = np.nanmax(envelope.envelope, axis=-1)
        t_end = envelope.t[np.argmax(envelope.envelope, axis=-1)]
        t_start = 0
        self.attackslope = (maximum - minimum) / (t_end - t_start)
        return self.attackslope