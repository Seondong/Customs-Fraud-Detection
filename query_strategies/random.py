import numpy as np
import random
from .strategy import Strategy

class RandomSampling(Strategy):
    """ Random strategy """
    
    def __init__(self, args):
        super(RandomSampling, self).__init__(args)

    def query(self, k):
        return np.random.choice(self.available_indices, k, replace = False).tolist()
