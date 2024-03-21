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
                child = parent.add_child(pipe)
                
                if(child.feature is None):
                    child.compute(parent)

                parent = child

            features.append(child.feature)
        self.print_node(self)
        return features

    def print_node(self, node, prefix=''):
        for child in node.children:
            print(prefix, child)
            self.print_node(child, prefix + '\t')