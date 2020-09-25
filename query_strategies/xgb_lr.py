import numpy as np
import random
import sys
sys.path.append("..")
from .strategy import Strategy
from xgboost import XGBClassifier
from generate_loader import separate_train_test_data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

        

class XGBLRSampling(Strategy):
    def __init__(self, model_path, test_data, test_loader, args):
        self.model_path = './intermediary/xgb_models/xgb_model-'+args.identifier+'.json'
        self.identifier = args.identifier
        super(XGBLRSampling, self).__init__(self.model_path, test_data, test_loader, args)
    
    
    def get_xgb_model(self):
        xgb_clf = XGBClassifier(n_estimators=100, max_depth=4,n_jobs=-1)
        xgb_clf.load_model(self.model_path)
        return xgb_clf
    
    
    def load_data(self):
        _, _, _, _, _, _,_, _, _, _, xgb_trainx, xgb_trainy, xgb_validx, xgb_validy, xgb_testx, _ = separate_train_test_data(self.identifier)
        return xgb_trainx, xgb_trainy, xgb_validx, xgb_testx
    
        
    def get_xgb_lr_output(self):
        xgb_clf = self.get_xgb_model()
        xgb_trainx, xgb_trainy, xgb_validx, xgb_testx = self.load_data()
        
        # get leaf index from xgboost model 
        X_train_leaves = xgb_clf.apply(xgb_trainx)
        X_valid_leaves = xgb_clf.apply(xgb_validx)
        X_test_leaves = xgb_clf.apply(xgb_testx)
        train_rows = X_train_leaves.shape[0]

        # one-hot encoding for leaf index
        xgbenc = OneHotEncoder(categories="auto")
        lr_trainx = xgbenc.fit_transform(X_train_leaves)
        
        # Todo: Can be refined by using validation set
        lr_validx = xgbenc.transform(X_valid_leaves)
        lr_testx = xgbenc.transform(X_test_leaves)
        
        # train logistic regression model 
        print("Training Logistic regression model...")
        lr = LogisticRegression(n_jobs=-1)
        lr.fit(lr_trainx, xgb_trainy)
        final_output = lr.predict_proba(lr_testx)[:,1]
        
        return final_output[self.available_indices]
        
        
    def query(self, k):
        output = self.get_xgb_lr_output()
        chosen = np.argpartition(output, -k)[-k:]
        return self.available_indices[chosen].tolist()