import numpy as np
import random
from .strategy import Strategy

class HybridSampling(Strategy):
	def __init__(self, model_path, test_loader, args, subsamps, weights):
		super(HybridSampling, self).__init__(model_path, test_loader, args)
		assert sum(weights) == 1
		assert len(subsamps) == len(weights)
		self.subsamps = subsamps
		self.weights = weights
	def query(self, k):
		ks = [round(k*weight) for weight in self.weights[:-1]]
		ks.append(k - sum(ks))
		chosen = []
		for subsamp, num_samp in zip(self.subsamps, ks):
			subsamp.set_available_indices(chosen)
			chosen = [*chosen, *subsamp.query(num_samp)]
		return chosen