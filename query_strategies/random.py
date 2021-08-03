import numpy as np
import random
from .strategy import Strategy
from utils import timer_func

class RandomSampling(Strategy):
    """ Random strategy """
    
    def __init__(self, args):
        super(RandomSampling, self).__init__(args)

    @timer_func
    def query(self, k):
        return np.random.choice(self.available_indices, k, replace = False).tolist()
