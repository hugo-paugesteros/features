import numpy as np
import numpy.typing as npt

from . import Attack

class Log():

    def __init__(self):
        pass

    def compute(self, feature) -> None:
        if isinstance(feature, Attack):
            return np.log10(feature.attack)