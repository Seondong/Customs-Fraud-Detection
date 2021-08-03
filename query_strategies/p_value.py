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
from .drift import DriftSampling
from scipy.stats import anderson_ksamp
from utils import timer_func

class pvalueSampling(DriftSampling):
    
    def __init__(self, args):
        super(pvalueSampling, self).__init__(args)
        assert len(self.subsamps) == 2
            
    def small_shift(self, xs, xt):
        xs = xs.data.cpu().numpy()
        xt = xt.data.cpu().numpy()
        
        result = []
        for i in range(16):
            lol = anderson_ksamp([list(xs[:, i]), list(xt[:, i])])[2]
            result.append(lol)
        
        return result
    
    def concept_drift(self):
        # Measure concept drift between validation data and test data.
    
        valid_embeddings, test_embeddings = self.generate_DATE_embeddings()

        stack = []
        
        for j in range(60):                  # 60
            num_sample_valid = min(len(valid_embeddings), 500)
            num_sample_test = min(len(test_embeddings), 500)

            ind_valid = torch.tensor(random.sample(range(len(valid_embeddings)), num_sample_valid)).cuda()       
            ind_test = torch.tensor(random.sample(range(len(test_embeddings)), num_sample_test)).cuda()

            xv = torch.index_select(valid_embeddings, 0, ind_valid)
            xt = torch.index_select(test_embeddings, 0, ind_test)

            stack.append(self.small_shift(xv, xt))
        
        xd = np.mean(stack, axis = 0) # smaller value means greater shift :|
        xd = (xd < 0.05).sum()/ 16 # 16 is the dimension 
        # xd = 1 - min(1, xd.mean()/0.1)
        return xd.item()
    

    @timer_func
    def query(self, k):
        # Drift sampler should measure the concept drift and update subsampler weights before the query selection is made. 
        self.update_subsampler_weights()
        super(pvalueSampling, self).query(k)
        return self.chosen

        
