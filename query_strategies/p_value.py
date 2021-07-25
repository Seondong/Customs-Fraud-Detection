import numpy as np
import random
import sys
import math
import ot
from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd
import torch
from .strategy import Strategy
from .DATE import DATESampling
from .hybrid import HybridSampling
from scipy.stats import anderson_ksamp

class pvalueSampling(HybridSampling):
    
    def __init__(self, args):
        super(pvalueSampling, self).__init__(args)
        assert len(self.subsamps) == 2
        
    def generate_DATE_embeddings(self):

        date_sampler = DATESampling(self.args)
        date_sampler.set_data(self.data)
        date_sampler.train_xgb_model()
        date_sampler.prepare_DATE_input()
        date_sampler.train_DATE_model()
        valid_embeddings = torch.stack(date_sampler.get_embedding_valid())  # Embeddings for validation data
        test_embeddings = torch.stack(date_sampler.get_embedding_test())         # Embeddings for test data

        return valid_embeddings, test_embeddings
    
    def small_shift(self, xs, xt):
        xs = xs.data.cpu().numpy()
        xt = xt.data.cpu().numpy()
        
        result = []
        for i in range(16):
            lol = anderson_ksamp([list(xs[:, i]), list(xt[:, i])])[2]
            result.append(lol)
        
        return result
    
    def domain_shift(self):
        # Measure domain shift between validation data and test data.
    
        valid_embeddings, test_embeddings = self.generate_DATE_embeddings()

        stack = []
        
        for j in range(60):                  # 60
            ind_valid = torch.tensor(random.sample(range(len(valid_embeddings)), 500)).cuda()       # 500
            ind_test = torch.tensor(random.sample(range(len(test_embeddings)), 500)).cuda()         # 500

            xv = torch.index_select(valid_embeddings, 0, ind_valid)
            xt = torch.index_select(test_embeddings, 0, ind_test)

            stack.append(self.small_shift(xv, xt))
        
        xd = np.mean(stack, axis = 0)

        return xd # smaller value means greater shift :|


    def update_subsampler_weights(self):  
        
        weight = (self.domain_shift() < 0.05).sum()/ 16 # 16 is the dimension 
        # weight = 0.95 ** ((self.domain_shift() < 0.05).sum())
        
        self.weight = round(weight, 2).item()

        self.weights = [1 - self.weight, self.weight]
        
        
