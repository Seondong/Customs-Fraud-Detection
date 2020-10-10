import numpy as np
import random
from .strategy import Strategy

class RandomSampling(Strategy):
    """ Random strategy """
    
    def __init__(self, data, args):
        super(RandomSampling, self).__init__(data, args)

    def query(self, k, **kwargs):
        return np.random.choice(self.available_indices, k, replace = False).tolist()
