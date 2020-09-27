import numpy as np
import random
import sys
import pickle
sys.path.append("..")
from .strategy import Strategy
from pytorch_tabnet.tab_model import TabNetClassifier
from generate_loader import separate_train_test_data


class TabnetSampling(Strategy):
    def __init__(self, model_path, test_data, test_loader, args):
        self.model_path = './intermediary/tn_models/tn_model-'+args.identifier+'.pkl'
        self.identifier = args.identifier
        super(TabnetSampling, self).__init__(self.model_path, test_data, test_loader, args)
    
    
    def get_tn_model(self):
        with open(self.model_path, 'rb') as f:
            tn_clf = pickle.load(f)
        return tn_clf
    
    
    def load_test_data(self):
        _, _, _, _, _, _,_, _, _, _, _, _, _, _, tn_testx, _ = separate_train_test_data(self.identifier)
        return tn_testx
    
    
    def get_tn_output(self):
        tn_clf = self.get_tn_model()
        tn_testx = self.load_test_data()
        final_output = tn_clf.predict_proba(tn_testx.values)[:,1]
        return final_output[self.available_indices]
        
        
    def query(self, k):
        output = self.get_tn_output()
        chosen = np.argpartition(output, -k)[-k:]
        return self.available_indices[chosen].tolist()