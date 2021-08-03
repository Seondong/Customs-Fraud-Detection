import numpy as np
import random
from .strategy import Strategy
from utils import timer_func

class RiskProfileSampling(Strategy):
    """ Naive Risk Profile Sampling strategy """
    

    def __init__(self, args):
        super(RiskProfileSampling, self).__init__(args)


    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        self.y_totalrisk = self.data.dftestx[[col for col in self.data.dftestx.columns if 'RiskH' in col]].sum(axis=1)
        
    @timer_func
    def query(self, k):
        self.predict_frauds()
        chosen = np.argpartition(self.y_totalrisk[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()