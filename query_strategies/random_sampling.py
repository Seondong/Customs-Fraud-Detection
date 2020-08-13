import numpy as np
import random

# class RandomSampling(Strategy):
# 	def __init__(self, model_path, test_loader, args):
# 		super(RandomSampling, self).__init__(self, model_path, test_loader, args)

# 	def query(self, k):
# 		num_data = test_loader.dataset.tensors[-1].shape[0]
# 		return random.sample(range(0,num_data-1),k)

class RandomSampling():
	def __init__(self, test_loader):
		self.test_loader = test_loader

	def query(self, k):
		num_data = test_loader.dataset.tensors[-1].shape[0]
		return random.sample(range(0,num_data-1),k)