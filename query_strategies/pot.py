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

            
class POTSampling(HybridSampling):
    """ Optimal Transport strategy: Using POT library to measure domain shift and control subsampler weights 
        Reference: https://pythonot.github.io/all.html?highlight=emd2#ot.emd2 """


    def __init__(self, args):
        super(POTSampling,self).__init__(args)
        assert len(self.subsamps) == 2
        
        self.intercept = -1.88325971450706   # SD: How to decide this value?
        self.coef = 0.00709741           # SD: How to decide this value? (Very small)

        # self.data already exists - In main.py, we declared in: sampler.set_data(data)
        
    
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
        return unnorm/inf # should be in range 0 and 1 :D Closer to 1 meaning more Domain Shift

 
    def generate_DATE_embeddings(self):

        date_sampler = DATESampling(self.args)
        date_sampler.set_data(self.data)
        date_sampler.train_xgb_model()
        date_sampler.prepare_DATE_input()
        date_sampler.train_DATE_model()
        valid_embeddings = torch.stack(date_sampler.get_embedding_valid())  # Embeddings for validation data
        test_embeddings = torch.stack(date_sampler.get_embedding_test())         # Embeddings for test data

        return valid_embeddings, test_embeddings


    def domain_shift(self):
        # Measure domain shift between validation data and test data.
    
        valid_embeddings, test_embeddings = self.generate_DATE_embeddings()
        stack = []
        
        # The code becomes slow when we control this number larger, need to optimize the calculation in 'small_shift'
        for j in range(50):                  # 60
            ind_valid = torch.tensor(random.sample(range(len(valid_embeddings)), 500)).cuda()       # 500
            ind_test = torch.tensor(random.sample(range(len(test_embeddings)), 500)).cuda()         # 500

            xv = torch.index_select(valid_embeddings, 0, ind_valid)
            xt = torch.index_select(test_embeddings, 0, ind_test)

            stack.append(self.small_shift2(xv, xt).item())

        xd = np.mean(stack)
        return xd


    def update_subsampler_weights(self):  
        weight = self.domain_shift()
        self.weight = round(weight, 2).item()
        self.weights = [1 - self.weight, self.weight]
        print("prob:", weight)




    