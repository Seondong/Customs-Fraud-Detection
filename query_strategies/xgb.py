import numpy as np
import random
import sys
sys.path.append("..")
from .strategy import Strategy
from xgboost import XGBClassifier
from generate_loader import separate_train_test_data


class XGBSampling(Strategy):
    def __init__(self, model_path, test_data, test_loader, args):
        self.model_path = './intermediary/xgb_model-'+args.identifier+'.json'
        self.identifier = args.identifier
        super(XGBSampling, self).__init__(self.model_path, test_data, test_loader, args)
    
    
    def get_xgb_model(self):
        xgb_clf = XGBClassifier(n_estimators=100, max_depth=4,n_jobs=-1)
        xgb_clf.load_model(self.model_path)
        return xgb_clf
    
    
    def load_test_data(self):
        _, _, _, _, _, _,_, _, _, _, _, _, _, _, xgb_testx, _ = separate_train_test_data(self.identifier)
        return xgb_testx
    
    
    def get_xgb_output(self):
        xgb_clf = self.get_xgb_model()
        xgb_testx = self.load_test_data()
        final_output = xgb_clf.predict_proba(xgb_testx)[:,1]
        return final_output[self.available_indices]
        
        
    def query(self, k):
        output = self.get_xgb_output()
        chosen = np.argpartition(output, -k)[-k:]
        return self.available_indices[chosen].tolist()