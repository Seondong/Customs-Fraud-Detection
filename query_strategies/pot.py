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
from .DATE import DATESampling
from .drift import DriftSampling

            
class POTSampling(DriftSampling):
    """ Optimal Transport strategy: Using POT library to measure concept drift and control subsampler weights 
        Reference: https://pythonot.github.io/all.html?highlight=emd2#ot.emd2 """


    def __init__(self, args):
        super(POTSampling,self).__init__(args)
        assert len(self.subsamps) == 2
        
    
    def small_shift(self, xs, xt):
        M = torch.cdist(xs, xt)
        lol = torch.max(M)
        M = M/lol
        M = M.data.cpu().numpy()
        a = [1/xs.shape[0]] * xs.shape[0]
        b = [1/xt.shape[0]] * xt.shape[0]
        prep = ot.emd2(a, b, M)
        unnorm = prep * lol
        return unnorm
    
    
    def small_shift2(self, xs, xt):
        M = torch.cdist(xs, xt)    
        lol = torch.max(M)        
        M = M/lol
        M = M.data.cpu().numpy()    
        a = [1/xs.shape[0]] * xs.shape[0]
        b = [1/xt.shape[0]] * xt.shape[0]    
        prep = ot.emd2(a, b, M)        
        unnorm = prep * lol
        
        xsnorm = torch.norm(xs, dim = 1)
        xssumnorm = xsnorm.sum()/xs.shape[0]        
        xtnorm = torch.norm(xt, dim = 1)
        xtsumnorm = xtnorm.sum()/xt.shape[0]        
        inf = xssumnorm + xtsumnorm        
        return unnorm/inf # should be in range 0 and 1 :D Closer to 1 meaning more concept drift


    def concept_drift(self):
        # Measure concept drift between validation data and test data.
    
        valid_embeddings, test_embeddings = self.generate_DATE_embeddings()
        stack = []
        
        # The code becomes slow when we control this number larger, need to optimize the calculation in 'small_shift'
        for j in range(50):                  # 60
            num_sample_valid = min(len(valid_embeddings), 500)
            num_sample_test = min(len(test_embeddings), 500)

            ind_valid = torch.tensor(random.sample(range(len(valid_embeddings)), num_sample_valid)).cuda()       
            ind_test = torch.tensor(random.sample(range(len(test_embeddings)), num_sample_test)).cuda()

            xv = torch.index_select(valid_embeddings, 0, ind_valid)
            xt = torch.index_select(test_embeddings, 0, ind_test)

            stack.append(self.small_shift2(xv, xt).item())

        xd = np.mean(stack)
        return xd.item()


    def query(self, k):
        # Drift sampler should measure the concept drift and update subsampler weights before the query selection is made. 
        self.update_subsampler_weights()
        super(POTSampling, self).query(k)
        return self.chosen




    