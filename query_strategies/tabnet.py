import numpy as np
import random
import sys
import pickle
sys.path.append("..")
from .strategy import Strategy
from pytorch_tabnet.tab_model import TabNetClassifier


class TabnetSampling(Strategy):
    """ TabNet strategy: Using TabNet classification probability to measure fraudness of imports 
        Code reference: https://github.com/dreamquark-ai/tabnet
        Paper reference: https://arxiv.org/pdf/1908.07442.pdf """
    
    
    def __init__(self, args):
        super(TabnetSampling, self).__init__(args)

    
    def train_model(self):
        print("Training TabNet model...")
        self.tn = TabNetClassifier()
#         self.tn.fit(xgb_trainx.values, xgb_trainy, xgb_validx.values, xgb_validy)
        self.tn.fit(self.data.X_train_lab, self.data.train_cls_label, self.data.X_valid_lab, self.data.valid_cls_label)
    
    
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        self.y_prob = self.tn.predict_proba(self.data.X_test)[:,1]
        

    def query(self, k):
        self.train_model()
        self.predict_frauds()
        chosen = np.argpartition(self.y_prob[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()