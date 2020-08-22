import numpy as np
import random
from .strategy import Strategy

class RandomSampling(Strategy):
	def __init__(self, model_path, test_loader, args):
		super(RandomSampling, self).__init__(model_path, test_loader, args)

	def query(self, k):
		return np.random.choice(self.available_indices, k, replace = False).tolist()
