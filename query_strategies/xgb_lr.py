import numpy as np
import random
import sys
sys.path.append("..")
from .strategy import Strategy
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

        
class XGBLRSampling(Strategy):
    """ XGBoost+LR strategy: Using Logistic Regression based on XGBoost leaf indices to measure fraudness of imports """
    
    def __init__(self, args):
        super(XGBLRSampling, self).__init__(args)
    
    
    def train_xgb_model(self):
        """ Train XGB model """
        print("Training XGBoost model...")
        self.xgb = XGBClassifier(n_estimators=100, max_depth=4, n_jobs=-1)
        self.xgb.fit(self.data.dftrainx_lab, self.data.train_cls_label)
        
        if self.args.save:
            self.xgb.get_booster().dump_model('./intermediary/xgb_models/xgb_model-readable-'+self.args.identifier+'.txt', with_stats=False)
            self.xgb.get_booster().save_model('./intermediary/xgb_models/xgb_model-'+self.args.identifier+'.json')
        
    
    def prepare_lr_input(self):
        """ Prepare input for logistic regression model """
        # Get leaf index from xgboost model 
        X_train_leaves = self.xgb.apply(self.data.dftrainx_lab)
        X_valid_leaves = self.xgb.apply(self.data.dfvalidx_lab)
        X_test_leaves = self.xgb.apply(self.data.dftestx)
        
        # One-hot encoding for leaf index
        # Todo: Can be refined by using validation set
        xgbenc = OneHotEncoder(categories="auto")
        self.lr_trainx = xgbenc.fit_transform(X_train_leaves)
        self.lr_validx = xgbenc.transform(X_valid_leaves)
        self.lr_testx = xgbenc.transform(X_test_leaves)
        
        
    def train_lr_model(self):
        """ Train LR model """
        print("Training Logistic Regression model...")
        self.lr = LogisticRegression(n_jobs=-1, max_iter=1000)
        self.lr.fit(self.lr_trainx, self.data.train_cls_label)
        
        
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        self.y_prob = self.lr.predict_proba(self.lr_testx)[:,1]
        
        
    def query(self, k):
        """ Querying top-k imports to inspect """
        self.train_xgb_model()
        self.prepare_lr_input()
        self.train_lr_model()
        self.predict_frauds()
        chosen = np.argpartition(self.y_prob[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()