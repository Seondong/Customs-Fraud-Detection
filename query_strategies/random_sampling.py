import numpy as np
import random
from .strategy import Strategy

class RandomSampling(Strategy):
	def __init__(self, model_path, test_loader, uncertainty_module, args):
		super(RandomSampling, self).__init__(model_path, test_loader, uncertainty_module, args)

	def query(self, k):
		num_data = self.test_loader.dataset.tensors[-1].shape[0]
		return random.sample(range(0,num_data-1),k)
