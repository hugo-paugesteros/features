import numpy as np
import numpy.typing as npt

from features.PipelineElement import PipelineElement

class Waveform(PipelineElement):

    y:  npt.NDArray
    sr: int
    t:  npt.NDArray

    def __init__(self, y: npt.NDArray, sr: int):
        super().__init__()
        
        self.y  = y
        self.sr = sr
        self.t  = np.arange(y.shape[-1]) / sr

    def compute(self):
        return self

    def extract(self, pipelines):
        features = []

        for pipeline in pipelines:
            parent = self
            for pipe in pipeline:
                pipe.PARENT = parent
                child = parent.add_child(pipe)
                
                if(child.feature is not None):
                    feature = child.feature
                else:
                    feature = child.compute(parent)
                parent = child

            features.append(feature)
        return features
