import pandas as pd
import numpy as np
import preprocess_data
from xgboost import XGBClassifier, XGBRegressor

def uncertainty_naive(test, column_to_use_unc_measure) :
    # Model 1 : Naive-equally contributing uncertainty

    uncertainty_coefficient = 1/(len(column_to_use_unc_measure))

    test['uncertainty_score'] = 0

    for col in column_to_use_unc_measure :
        test['uncertainty_score'] = test['uncertainty_score'] + uncertainty_coefficient * test['unc.'+col]
    
    return test['uncertainty_score']

def scaling(x) :
    if x < 0 :
        return 0
    elif x > 1 :
        return 1
    return x

def uncertainty_measurement(train, test) :
    # Columns to use
    numeric_columns = [fob.value', 'cif.value', 'total.taxes', 'gross.weight', 'quantity', 'Unitprice', 'WUnitprice', 'TaxRatio', 'FOBCIFRatio', 'TaxUnitquantity', 'SGD.DayofYear', 'SGD.WeekofYear', 'SGD.MonthofYear']
    categorical_columns = [col for col in train.columns if 'RiskH' in col]
    column_to_use_unc_measure = numeric_columns + categorical_columns

    train_unc = pd.DataFrame(train, columns = column_to_use_unc_measure)
    test_unc = pd.DataFrame(test, columns = column_to_use_unc_measure)

    # Generate classifiers for predict each masked categorical feature
    for cc in categorical_columns :
        columns = [col for col in column_to_use_unc_measure if col != cc]

        print("Training xgboost model for predicting %s" % cc)
        xgb_trainx = pd.DataFrame(train, columns = columns)
        xgb_testx = pd.DataFrame(test, columns = columns)

        xgb_clf = XGBClassifier(n_estimators=100)
        xgb_clf.fit(xgb_trainx,train[cc].values)

        xgb_clf_pred = xgb_clf.predict(xgb_testx)
        test['pred.'+cc] = xgb_clf_pred.tolist()
        test['unc.'+cc] = np.bitwise_xor(test[cc], test['pred.'+cc])

    # Generate regressors for predict each masked numeric feature
    for nc in numeric_columns :
        columns = []
        columns = [col for col in column_to_use_unc_measure if col != nc]
        
        print("Training xgboost model for predicting %s" % nc)
        xgb_trainx = pd.DataFrame(train, columns=columns)
        xgb_testx = pd.DataFrame(test, columns=columns)
        
        # Regression Model
        xgb_reg = XGBRegressor(n_estimators = 100)
        xgb_reg.fit(xgb_trainx, train[nc].values)
        
        xgb_reg_pred = xgb_reg.predict(xgb_testx)
        test['pred.'+nc] = xgb_reg_pred.tolist()
        test['unc.'+nc] = scaling(abs(test[nc] - test['pred.'+nc]) / test[nc])
        print(test['pred.'+nc].describe())
    
    return uncertainty_naive(test, column_to_use_unc_measure)