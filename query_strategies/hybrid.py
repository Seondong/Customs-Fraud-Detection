import numpy as np
import random
from .strategy import Strategy
from .DATE import DATESampling

class HybridSampling(Strategy):
    def __init__(self, data, args, subsamps, weights):
        super(HybridSampling, self).__init__(data, args)
        assert sum(weights) == 1
        assert len(subsamps) == len(weights)
        self.subsamps = subsamps
        self.weights = weights
#         self.available_indices = None  # Needed!

    def query(self, k):
        self.ks = [round(k*weight) for weight in self.weights[:-1]]
        self.ks.append(k - sum(self.ks))
        chosen = []
        trained_DATE_available = False
        for subsamp, num_samp in zip(self.subsamps, self.ks):
            print(f'Querying {num_samp} items for subsampler {subsamp}')
            subsamp.set_available_indices(chosen)
            if isinstance(subsamp, DATESampling):
                chosen = [*chosen, *subsamp.query(num_samp, model_available = trained_DATE_available)]
                trained_DATE_available = True
            else:
                chosen = [*chosen, *subsamp.query(num_samp)]
        return chosen