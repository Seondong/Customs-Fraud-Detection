import numpy as np
import random
import sys
from .strategy import Strategy
from .DATE import DATESampling
from main import initialize_sampler

class HybridSampling(Strategy):
    def __init__(self, args):
        super(HybridSampling, self).__init__(args)
        self.subsamps = [initialize_sampler(subsamp, args) for subsamp in args.subsamplings.split("/")] 
        self.weights = [float(weight) for weight in args.weights.split("/")]
        assert round(sum(self.weights), 10) == 1
        assert len(self.subsamps) == len(self.weights)
#         self.available_indices = None  # Needed!
     
    
    def set_data(self, data):
        super(HybridSampling, self).set_data(data)
        for subsamp in self.subsamps:
            subsamp.set_data(data)
    
    
    def set_weights(self, weights):
        self.weights = weights
        
    
    def get_weights(self):
        return self.weights
    
    
    def set_uncertainty_module(self, uncertainty_module):
        super(HybridSampling, self).set_uncertainty_module(uncertainty_module)
        for subsamp in self.subsamps:
            subsamp.uncertainty_module = uncertainty_module
        
        
    def query(self, k):
        self.ks = [round(k*weight) for weight in self.weights[:-1]]
        self.ks.append(k - sum(self.ks))
        self.chosen = []
        self.each_chosen = {}
        trained_DATE_available = False
        for subsamp, num_samp in zip(self.subsamps, self.ks):
            if num_samp == 0:
                continue
            print(f'<Hybrid> Querying {num_samp} (={round(100*num_samp/np.sum(self.ks))}%) items using the {subsamp} subsampler')
            subsamp.set_available_indices(self.chosen)
            if isinstance(subsamp, DATESampling):
                self.chosen = [*self.chosen, *subsamp.query(num_samp, model_available = trained_DATE_available)]
                self.each_chosen[subsamp.name] = subsamp.query(num_samp, model_available = trained_DATE_available) 
                trained_DATE_available = True
            else:
                self.chosen = [*self.chosen, *subsamp.query(num_samp)]
                self.each_chosen[subsamp.name] = subsamp.query(num_samp) 
        return self.chosen