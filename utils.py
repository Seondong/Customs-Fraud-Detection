from sklearn.metrics import f1_score,roc_auc_score
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
import pandas as pd 
from scipy.stats import hmean
from time import time
from sklearn.preprocessing import MultiLabelBinarizer


def timer_func(func):
    """
    This function shows the execution time of the function object passed
    """
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        # print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')   # You can turn it on/off for debugging.
        return result
    return wrap_func


def find_best_threshold(model,x_list,y_test,best_thresh = None):
    '''
    dtype model: scikit-learn classifier model
    dtype x_list: list or array to predict the probability result
    dtype y_test: array of true labels
    
    Find the best probability threshold to separate probability to 0 and 1
    '''
    y_prob = model.predict_proba(x_list)[:,1]
    threshold_list = np.arange(0.1,0.6,0.1)
    best_auc = 0.5    # 0.5 is random for AUC.
    
    if best_thresh ==None:
        for th in threshold_list:
            y_pred_label = (y_prob > th)*1 
            try:
                auc_score = roc_auc_score(y_test,y_prob)
            except ValueError:
                auc_score = 0.5
            if auc_score > best_auc:
                best_auc = auc_score
                best_thresh = th 
        return best_thresh, best_auc
    
    else:
        y_pred_label = (y_prob > best_thresh)*1 
        best_auc = roc_auc_score(y_test,y_pred_label)
    print("AUC-score equals to:%.4f"%(best_auc))
    return best_auc


def torch_threshold(y_prob,y_test,best_thresh = None):
    threshold_list = np.arange(0.1,0.6,0.1)
    best_f1 = 0
    if best_thresh == None:
        for th in threshold_list:
            y_pred_label = (y_prob > th)*1 
            f_score = f1_score(y_test[~np.isnan(y_test)],y_pred_label[~np.isnan(y_test)])
            if f_score > best_f1:
                best_f1 = f_score
                best_thresh = th 
        try:
            roc_auc = roc_auc_score(y_test[~np.isnan(y_test)], y_prob[~np.isnan(y_test)])
        except ValueError:
            roc_auc = 0.5
        return best_thresh, best_f1, roc_auc
    
    else:
        y_pred_label = (y_prob > best_thresh)*1 
        best_f1 = f1_score(y_test[~np.isnan(y_test)],y_pred_label[~np.isnan(y_test)])
        try:
            roc_auc = roc_auc_score(y_test[~np.isnan(y_test)], y_prob[~np.isnan(y_test)])
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


def metrics(y_prob,y_cls,y_rev, args, best_thresh=None):
    """ Evaluate the performance"""
    if best_thresh == None:
        _,overall_f1,auc = torch_threshold(y_prob,y_cls,best_thresh)
    else:
        overall_f1,auc = torch_threshold(y_prob,y_cls,best_thresh)
    pr, re, f, rev = [], [], [], []
    # For validatation, we measure the performance on 5% (previously, 1%, 2%, 5%, and 10%)
    for i in [95]: 
        threshold = np.percentile(y_prob, i)
        precision = y_cls[y_prob > threshold].mean()
        recall = sum(y_cls[y_prob > threshold])/ sum(y_cls)
        try:
            f1 = hmean([precision, recall])
        except ValueError:
            f1 = 0
        revenue = sum(y_rev[y_prob > threshold]) / sum(y_rev)
        # if i == 95:
        #     print(f'Checking top {100-i}% suspicious transactions: {len(y_prob[y_prob > threshold])}')
        #     print('Precision: %.4f, Recall: %.4f, Revenue: %.4f' % (precision, recall, revenue))
        pr.append(precision)
        re.append(recall)
        f.append(f1)
        rev.append(revenue)
    return overall_f1,auc,pr, re, f, rev


def evaluate_inspection(chosen_rev,chosen_cls,y_cls,y_rev):
    """ Evaluate the model performance """
    try:
        precision = np.count_nonzero(chosen_cls == 1) / len(chosen_cls)
    except:
        precision = np.float("nan")
    try:
        recall = sum(chosen_cls) / sum(y_cls)
    except:
        recall = np.float("nan")
    try:
        f1 = hmean([precision, recall])
    except ValueError:
        f1 = np.float("nan")
    try:
        revenue_avg = sum(chosen_rev)/len(chosen_cls)
    except:
        revenue_avg = np.float("nan")
    try:
        revenue_recall = sum(chosen_rev) / sum(y_rev)
    except ZeroDivisionError:
        revenue_recall = np.float("nan")
    return precision, recall, f1, revenue_avg, revenue_recall


def evaluate_inspection_multiclass(inspected, test, class_labels):
    """ Evaluate the model performance - for kdata (multi-class, multi-label datasets)"""

    inspection_codes = class_labels['검사결과부호']
    inspection_codes_broad = sorted(list(set(class_labels['검사결과부호'].apply(lambda x: x[0]))))
    result = {}

    @timer_func
    def _calculate_metrics(codes, label):
        mlb = MultiLabelBinarizer(classes = list(range(len(codes))))
        iresults = inspected[label]
        tresults = test[label]
        iresults_mtx = np.array(mlb.fit_transform(iresults)) # change into matrix..
        tresults_mtx = np.array(mlb.fit_transform(tresults))

        precisions = np.true_divide(iresults_mtx.sum(axis = 0), np.shape(iresults_mtx)[0]) # array of precisions
        recalls = np.divide(iresults_mtx.sum(axis = 0), tresults_mtx.sum(axis = 0), out = np.zeros(len(codes)), where = tresults_mtx.sum(axis = 0)!=0) # array of recalls
        try:
            f1 = hmean([precisions, recalls], axis = 0)
        except ValueError:
            f1 = np.zeros(len(precisions))
        macro_f1 = np.mean(np.array(f1))
        result['precision'] = dict(zip(codes, precisions))
        result['recall'] = dict(zip(codes, recalls))
        result['f1'] = dict(zip(codes, f1))
        result['macrof1'] = macro_f1
        return result

    result['specific_result'] = _calculate_metrics(inspection_codes, '검사결과코드') 
    result['broad_result'] = _calculate_metrics(inspection_codes_broad, '검사결과코드-대분류') 
    return result