import numpy as np
from torch.utils.data import DataLoader
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import deepcopy
from strategy import Strategy

class DATESampling(Strategy):
    def __init__(self, model_path, test_loader, uncertainty_module, args):
        super(DATESampling,self).__init__(model_path, test_loader, uncertainty_module, args)

    def query(self, k):
        output = self.get_output()
        chosen = np.argpartition(output, -k)[-k:]
        return chosen

