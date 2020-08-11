import pandas as pd
import numpy as np
import preprocess_data
import pickle
from xgboost import XGBClassifier, XGBRegressor
from utils import find_best_threshold
from sklearn.metrics import f1_score,roc_auc_score

# Columns to use
numeric_columns = ['fob.value', 'cif.value', 'total.taxes', 'gross.weight', 'quantity', 'Unitprice', 'WUnitprice', 'TaxRatio', 'FOBCIFRatio', 'TaxUnitquantity']
categorical_columns = ['RiskH.importer.id', 'RiskH.declarant.id',
    'RiskH.HS6.Origin', 'RiskH.tariff.code', 'RiskH.HS6',
    'RiskH.HS4', 'RiskH.HS2', 'RiskH.office.id']
column_to_use_unc_measure = numeric_columns + categorical_columns

def scaling(x) :
    if x < 0 :
        return 0
    elif x > 1 :
        return 1
    return x

def uncertainty_tag(train, test, unc, option) :
    test['uncertain'] = 0

    if option == 'naive' :
        # Model 1 : Naive equally-contributing uncertainty (mean)
        test['uncertain'] = unc.mean(axis=1)

    elif option == 'feature_importance' :
        # Model 2 : Feature importance from illicitness
        xgb_illicit = XGBClassifier(n_estimators = 100)
        xgb_illicit.fit(pd.DataFrame(train, columns=column_to_use_unc_measure), pd.DataFrame(train, columns=['illicit']))
        test['uncertain'] = unc.dot(xgb_illicit.feature_importances_)

    # elif option == 'fip_weighted' :
    #     # Model 3 : Frequency inverse proportional weighted sum
    #     train_using = train['unc.'+column_to_use_unc_measure]
    #     fip_weight = train_using.sum() / len(train_using.columns)
    #     test['uncertain'] = unc.dot(fip_weight)

    else :
        print('uncertainty_tag : Invalid option')

def uncertainty_measurement(train, valid, test, option) :
    unc = pd.DataFrame()

    train_unc = pd.DataFrame(train, columns = column_to_use_unc_measure)
    valid_unc = pd.DataFrame(valid, columns = column_to_use_unc_measure)
    test_unc = pd.DataFrame(test, columns = column_to_use_unc_measure)

    # Generate classifiers for predict each masked categorical feature
    for cc in categorical_columns :
        columns = [col for col in column_to_use_unc_measure if col != cc]

        print("Training xgboost model for predicting %s" % cc)
        xgb_trainx = pd.DataFrame(train_unc, columns = columns)
        xgb_validx = pd.DataFrame(valid_unc, columns = columns)
        xgb_testx = pd.DataFrame(test_unc, columns = columns)

        xgb_clf = XGBClassifier(n_estimators=100)
        xgb_clf.fit(xgb_trainx,train[cc].values)

        # evaluate xgboost model
        print("------Evaluating xgboost model------")
        test_pred = xgb_clf.predict_proba(xgb_testx)[:,1]
        xgb_auc = roc_auc_score(test[cc].values, test_pred)
        xgb_threshold,_ = find_best_threshold(xgb_clf, xgb_validx, valid[cc].values)
        xgb_f1 = find_best_threshold(xgb_clf, xgb_testx, test[cc].values,best_thresh=xgb_threshold)
        print("AUC = %.4f, F1-score = %.4f" % (xgb_auc, xgb_f1))
        print("------------------------------------")

        xgb_clf_pred = xgb_clf.predict(xgb_testx)
        unc['unc.'+cc] = np.bitwise_xor(test[cc], xgb_clf_pred.tolist())

        # Descriptive statistical analysis
        print("------Descriptive statistics------")
        print(pd.Categorical(xgb_clf_pred.tolist()).value_counts())
        print("----------------------------------")

        # For unseen characters, mark them as uncertain case 
        for idx, cat in enumerate(test[cc[6:]]) :
            if cat not in train[cc[6:]] :
                unc['unc.'+cc][idx] = 1

    # Generate regressors for predict each masked numeric feature
    for nc in numeric_columns :
        columns = []
        columns = [col for col in column_to_use_unc_measure if col != nc]
        
        print("Training xgboost model for predicting %s" % nc)
        xgb_trainx = pd.DataFrame(train_unc, columns=columns)
        xgb_testx = pd.DataFrame(test_unc, columns=columns)
        
        # Regression Model
        xgb_reg = XGBRegressor(n_estimators = 100)
        xgb_reg.fit(xgb_trainx, train[nc].values)
        
        xgb_reg_pred = xgb_reg.predict(xgb_testx)
        unc['unc.'+nc] = abs(test[nc] - xgb_reg_pred.tolist()) / test[nc]
        unc['unc.'+nc] = unc['unc.'+nc].apply(lambda x : scaling(x))

        # Descriptive statistical anaylsis
        print("------Descriptive statistics------")
        print(pd.DataFrame(xgb_reg_pred.tolist()).describe())
        print("----------------------------------")
    
    uncertainty_tag(train, test, unc, option)

# -----------------------------------------------
# Temporary test codes
# -----------------------------------------------

# load preprocessed data
with open("./processed_data.pickle","rb") as f :
    processed_data = pickle.load(f)
print(processed_data.keys())
print("Finish loading data...")

# train/test data 
train = processed_data["raw"]["train"]
valid = processed_data["raw"]["valid"]
test = processed_data["raw"]["test"]

uncertainty_measurement(train, valid, test, 'feature_importance')
print(test.columns)
print(test.sort_values('uncertain', ascending=False)[1:20])
