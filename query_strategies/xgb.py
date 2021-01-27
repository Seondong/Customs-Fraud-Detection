import numpy as np
import random
import sys
sys.path.append("..")
from .strategy import Strategy
from xgboost import XGBClassifier


class XGBSampling(Strategy):
    """ XGBoost strategy: Using XGB classification probability to measure fraudness of imports """
    
    def __init__(self, args):
        super(XGBSampling, self).__init__(args)
    
    
    def train_model(self):
        """ Get trained model """
        print("Training XGBoost model...")
        self.xgb = XGBClassifier(n_estimators=100, max_depth=4, n_jobs=-1)
        self.xgb.fit(self.data.dftrainx_lab,self.data.train_cls_label)
        
        if self.args.save:
            self.xgb.get_booster().dump_model('./intermediary/xgb_models/xgb_model-readable-'+self.args.identifier+'.txt', with_stats=False)
            self.xgb.get_booster().save_model('./intermediary/xgb_models/xgb_model-'+self.args.identifier+'.json')
        
    
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        self.y_prob = self.xgb.predict_proba(self.data.dftestx)[:,1]
        
        
        
    def query(self, k):
        self.train_model()
        self.predict_frauds()
        chosen = np.argpartition(self.y_prob[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()
    
    
    
    
    ############ Code block for further debugging
    #      def _run_XGB():
        
#         self.model = XGBClassifier(n_estimators=100, max_depth=4,n_jobs=-1)
#         self.model.fit(xgb_trainx,xgb_trainy)
        
#         # evaluate xgboost model
#         print("------Evaluating xgboost model------")
#         test_pred = self.model.predict_proba(xgb_testx)[:,1]
#         xgb_auc = roc_auc_score(xgb_testy, test_pred)
#         xgb_threshold,_ = find_best_threshold(self.model, xgb_validx, xgb_validy)
#         xgb_f1 = find_best_threshold(self.model, xgb_testx, xgb_testy,best_thresh=xgb_threshold)
#         print("AUC = %.4f, F1-score = %.4f" % (xgb_auc, xgb_f1))

#         # Precision and Recall
#         y_prob = test_pred
#         for i in [99,98,95,90]:
#             threshold = np.percentile(y_prob, i)
#             print(f'Checking top {100-i}% suspicious transactions: {len(y_prob[y_prob > threshold])}')
#             precision = np.mean(xgb_testy[y_prob > threshold])
#             recall = sum(xgb_testy[y_prob > threshold])/sum(xgb_testy)
#             revenue_recall = sum(revenue_test[y_prob > threshold]) /sum(revenue_test)
#             print(f'Precision: {round(precision, 4)}, Recall: {round(recall, 4)}, Seized Revenue (Recall): {round(revenue_recall, 4)}')

#         self.model.get_booster().dump_model('./intermediary/xgb_models/xgb_model-readable-'+curr_time+'.txt', with_stats=False)
#         self.model.get_booster().save_model('./intermediary/xgb_models/xgb_model-'+curr_time+'.json')
 
#         return self.model


#         self.model_path = './intermediary/xgb_models/xgb_model-'+args.identifier+'.json'