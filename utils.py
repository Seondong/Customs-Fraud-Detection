from sklearn.metrics import f1_score,roc_auc_score
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
import pandas as pd 
from scipy.stats import hmean


def find_best_threshold(model,x_list,y_test,best_thresh = None):
    '''
    dtype model: scikit-learn classifier model
    dtype x_list: list or array to predict the probability result
    dtype y_test: array of true labels
    
    Find the best probability threshold to separate probability to 0 and 1
    '''
    y_pred_prob = model.predict_proba(x_list)[:,1]
    threshold_list = np.arange(0.1,0.6,0.1)
    best_auc = 0.5    # 0.5 is random for AUC.
    
    if best_thresh ==None:
        for th in threshold_list:
            y_pred_label = (y_pred_prob > th)*1 
            try:
                auc_score = roc_auc_score(y_test,y_pred_prob)
            except ValueError:
                auc_score = 0.5
            if auc_score > best_auc:
                best_auc = auc_score
                best_thresh = th 
        return best_thresh, best_auc
    
    else:
        y_pred_label = (y_pred_prob > best_thresh)*1 
        best_auc = roc_auc_score(y_test,y_pred_label)
    print("AUC-score equals to:%.4f"%(best_auc))
    return best_auc


def torch_threshold(y_pred_prob,y_test,best_thresh = None):
    threshold_list = np.arange(0.1,0.6,0.1)
    best_f1 = 0
    
    if best_thresh == None:
        for th in threshold_list:
            y_pred_label = (y_pred_prob > th)*1 
            f_score = f1_score(y_test,y_pred_label)
            if f_score > best_f1:
                best_f1 = f_score
                best_thresh = th 
        try:
            roc_auc = roc_auc_score(y_test, y_pred_prob)
        except ValueError:
            roc_auc = 0.5
        return best_thresh, best_f1, roc_auc
    
    else:
        y_pred_label = (y_pred_prob > best_thresh)*1 
        best_f1 = f1_score(y_test,y_pred_label)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_prob)
        except ValueError:
            roc_auc = 0.5
        return best_f1, roc_auc    

    
def process_leaf_idx(X_leaves): 
    '''
    Since the xgboost output represent leaf index for each tree
    We need to calculate total amount of leaves and assign unique index to each leaf
    Assign unique index for each leaf 
    '''
    leaves = X_leaves.copy()
    new_leaf_index = dict() # dictionary to store leaf index
    total_leaves = 0
    for c in range(X_leaves.shape[1]): # iterate for each column
        column = X_leaves[:,c]
        unique_vals = list(sorted(set(column)))
        new_idx = {v:(i+total_leaves) for i,v in enumerate(unique_vals)}
        for i,v in enumerate(unique_vals):
            leaf_id = i+total_leaves
            new_leaf_index[leaf_id] = {c:v}
        leaves[:,c] = [new_idx[v] for v in column]
        total_leaves += len(unique_vals)
        
    assert leaves.ravel().max() == total_leaves - 1
    return leaves,total_leaves,new_leaf_index


def stratify_sample(y,test_size=0.2,seed=0):
    y_ser = pd.Series(y)
    y_pos = y_ser[y_ser==1]
    y_neg = y_ser[y_ser==0]
    test_pos_idx = y_pos.sample(frac=test_size,random_state=seed).index
    test_neg_idx = y_neg.sample(frac=test_size,random_state=seed).index
    test_idx = np.hstack((test_pos_idx,test_neg_idx))
    train_idx = np.array([idx for idx in range(y_ser.shape[0]) if idx not in test_idx])
    return train_idx, test_idx


def metrics(y_prob,xgb_testy,revenue_test,best_thresh=None):
    if best_thresh ==None:
        _,overall_f1,auc = torch_threshold(y_prob,xgb_testy,best_thresh)
    else:
        overall_f1,auc = torch_threshold(y_prob,xgb_testy,best_thresh)
#     import pdb
#     pdb.set_trace()
    pr, re, f, rev = [], [], [], []
    for i in [99,98,95,90]:
        threshold = np.percentile(y_prob, i)
        precision = xgb_testy[y_prob > threshold].mean()
        recall = sum(xgb_testy[y_prob > threshold])/ sum(xgb_testy)
        try:
            f1 = hmean([precision, recall])
        except ValueError:
            f1 = 0
            
        revenue = sum(revenue_test[y_prob > threshold]) / sum(revenue_test)
        print(f'Checking top {100-i}% suspicious transactions: {len(y_prob[y_prob > threshold])}')
        print('Precision: %.4f, Recall: %.4f, Revenue: %.4f' % (precision, recall, revenue))
        # save results
        pr.append(precision)
        re.append(recall)
        f.append(f1)
        rev.append(revenue)
    return overall_f1,auc,pr, re, f, rev


def metrics_active(active_rev,active_cls,xgb_testy,revenue_test):
    precision = np.count_nonzero(active_cls == 1) / len(active_cls)
    recall = sum(active_cls) / sum(xgb_testy)
    try:
        f1 = hmean([precision, recall])
    except ValueError:
        f1 = 0
    revenue = sum(active_rev) / sum(revenue_test)
    print()
    print("--------Evaluating the model---------")
    print('Precision: %.4f, Recall: %.4f, Revenue: %.4f' % (precision, recall, revenue))
    return precision, recall, f1, revenue