import numpy as np
import random
import sys
import math
from .DATE import DATESampling
from .strategy import Strategy
from .hybrid import HybridSampling
import ot
from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd
import torch
import random


class pot_shift(object):
    
    def __init__(self, test_start_1, test_end_1, test_start_2, test_end_2):
        self.test_start_1 = test_start_1
        self.test_end_1 = test_end_1
        self.test_start_2 = test_start_2
        self.test_end_2 = test_end_2
        
        data = pd.read_csv('.//query_strategies//pot_resources//tdata.csv')
        self.data = data[(data['sgd.date'] >= '16-07-01') & (data['sgd.date'] < '18-06-01')]
        
        self.data.reset_index(drop=True, inplace=True)
    
    def concat(self, test_start, test_end):
        
        with open(".//query_strategies/pot_resources/embd_0_3.pickle","rb") as f :
            processed_data = pickle.load(f)

        data0 = self.data[(self.data['sgd.date'] >= test_start) & (self.data['sgd.date'] < test_end)]
        start = data0.index[0]
        end = data0.index[-1]
        xd = []
        
        datelist = list(processed_data.keys())
        
        for j in range(start, end + 1):
            xd.append(processed_data[datelist[j]].reshape(1, -1))

        array1 = torch.cat(xd, axis=0)
        return(array1)
    
    def small_shift(self, xs, xt):
        
        M = np.zeros(shape = (xs.shape[0], xt.shape[0]))
        for i in range(xs.shape[0]):
            for j in range(xt.shape[0]):
                M[i][j] = np.square(torch.sum((xs[i] - xt[j]) ** 2).item())
        lol = np.max(M)
        M = M/np.max(M)
        a = [1/xs.shape[0]] * xs.shape[0]
        b = [1/xt.shape[0]] * xt.shape[0]
        prep = ot.emd2(a, b, M)
        return prep * lol
    
    def domain_shift(self):
        
        array1 = self.concat(self.test_start_1, self.test_end_1)
        array2 = self.concat(self.test_start_2, self.test_end_2)
        
        stack = []
        
        for j in range(60):
        
            indices = torch.tensor(random.sample(range(array1.shape[0]), 500))
            indices2 = torch.tensor(random.sample(range(array2.shape[0]), 500))

            xs = array1[indices]
            xt = array2[indices2]

            stack.append(self.small_shift(xs, xt))
        
        xd = np.mean(stack)
        return xd
        

    
class potSampling(HybridSampling):
    
    def __init__(self, args):
        super(potSampling,self).__init__(args)
        assert len(self.subsamps) == 2
        
        self.intercept = -1.88325971450706
        self.coef = 0.00709741
        
        
    def update(self, test_start_day, test_end_day, test_end_end):
        
        test_start = test_start_day.strftime('%y-%m-%d')
        test_end_1 = test_end_day.strftime('%y-%m-%d')
        test_end_2 = test_end_end.strftime('%y-%m-%d')
        
        lol = pot_shift(test_start, test_end_1, test_end_1, test_end_2)
        
        domshift = lol.domain_shift()
        
        weight = np.exp(self.intercept + self.coef * domshift)/ (1 + np.exp(self.intercept + self.coef * domshift))
        
        self.weight = round(weight, 2).item()
        print(type(self.weight))
        
        print(self.weight)
        self.weights = [1 - self.weight, self.weight]
        
        #self.weights = [1 - 0.14, 0.14]


        