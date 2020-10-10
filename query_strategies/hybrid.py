import numpy as np
import random
from .strategy import Strategy

class HybridSampling(Strategy):
	def __init__(self, data, args, subsamps, weights):
		super(HybridSampling, self).__init__(data, args)
		assert sum(weights) == 1
		assert len(subsamps) == len(weights)
		self.subsamps = subsamps
		self.weights = weights
		
	def query(self, k):
		self.ks = [round(k*weight) for weight in self.weights[:-1]]
		self.ks.append(k - sum(self.ks))
		chosen = []
		for subsamp, num_samp in zip(self.subsamps, self.ks):
			subsamp.set_available_indices(chosen)
			chosen = [*chosen, *subsamp.query(num_samp)]
		return chosen