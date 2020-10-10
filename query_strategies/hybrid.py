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
		
	def query(self, k, **kwargs):
		self.ks = [round(k*weight) for weight in self.weights[:-1]]
		self.ks.append(k - sum(self.ks))
		chosen = []
		trained_DATE_available = False
		for subsamp, num_samp in zip(self.subsamps, self.ks):			
			subsamp.set_available_indices(chosen)
			chosen = [*chosen, *subsamp.query(num_samp, DATE = trained_DATE_available)]
			if isinstance(subsamp, DATESampling):
				trained_DATE_available = True				
		return chosen