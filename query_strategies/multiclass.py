import numpy as np
import random
import sys
sys.path.append("..")
from .strategy import Strategy
from xgboost import XGBClassifier
from utils import timer_func


class MulticlassSampling(Strategy):
    """ Multiclass strategy: Using some model to find multiple types of frauds """
    
    def __init__(self, args):
        super(MulticlassSampling, self).__init__(args)
    
    
    
    def train_model(self):
        """ Get trained model """
        pass


    
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        # self.y_prob = 'blah'
        # self.y_prob = self.xgb.predict_proba(self.data.dftestx)[:,1]
        
        
        
    def query(self, k):
        self.train_model()
        self.predict_frauds()
        chosen = np.argpartition(self.y_prob[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()

    # # Random으로 테스트하고 싶으면 
    # @timer_func
    # def query(self, k):
    #     return np.random.choice(self.available_indices, k, replace = False).tolist()