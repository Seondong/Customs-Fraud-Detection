import pandas as pd
import numpy as np
import preprocess_data
import pickle
from xgboost import XGBClassifier, XGBRegressor

def uncertainty_tag(test, column_to_use_unc_measure, option) :
    test['uncertain'] = 0

    if option == 'naive' :
    # Model 1 : Naive equally-contributing uncertainty (mean)
        uncertainty_coefficient = 1/(len(column_to_use_unc_measure))

        for col in column_to_use_unc_measure :
            test['uncertain'] = test['uncertain'] + uncertainty_coefficient * test['unc.'+col]

def scaling(x) :
    if x < 0 :
        return 0
    elif x > 1 :
        return 1
    return x

def uncertainty_measurement(train, test) :
    # Columns to use
    # numeric_columns = ['fob.value', 'cif.value', 'total.taxes', 'gross.weight', 'quantity', 'Unitprice', 'WUnitprice', 'TaxRatio', 'FOBCIFRatio', 'TaxUnitquantity', 'SGD.DayofYear', 'SGD.WeekofYear', 'SGD.MonthofYear']
    numeric_columns = ['fob.value', 'cif.value', 'total.taxes', 'gross.weight', 'quantity', 'Unitprice', 'WUnitprice', 'TaxRatio', 'FOBCIFRatio', 'TaxUnitquantity']
    categorical_columns = ['RiskH.importer.id', 'RiskH.declarant.id',
       'RiskH.HS6.Origin', 'RiskH.tariff.code', 'RiskH.quantity', 'RiskH.HS6',
       'RiskH.HS4', 'RiskH.HS2', 'RiskH.office.id']
    column_to_use_unc_measure = numeric_columns + categorical_columns

    train_unc = pd.DataFrame(train, columns = column_to_use_unc_measure)
    test_unc = pd.DataFrame(test, columns = column_to_use_unc_measure)

    # Generate classifiers for predict each masked categorical feature
    for cc in categorical_columns :
        columns = [col for col in column_to_use_unc_measure if col != cc]

        print("Training xgboost model for predicting %s" % cc)
        xgb_trainx = pd.DataFrame(train_unc, columns = columns)
        xgb_testx = pd.DataFrame(test_unc, columns = columns)

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
        xgb_trainx = pd.DataFrame(train_unc, columns=columns)
        xgb_testx = pd.DataFrame(test_unc, columns=columns)
        
        # Regression Model
        xgb_reg = XGBRegressor(n_estimators = 100)
        xgb_reg.fit(xgb_trainx, train[nc].values)
        
        xgb_reg_pred = xgb_reg.predict(xgb_testx)
        test['pred.'+nc] = xgb_reg_pred.tolist()
        test['unc.'+nc] = abs(test[nc] - test['pred.'+nc]) / test[nc]
        test['unc.'+nc] = test['unc.'+nc].apply(lambda x : scaling(x))
        print(test['pred.'+nc].describe())
    
    uncertainty_tag(test, column_to_use_unc_measure, 'naive')

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

uncertainty_measurement(train, test)
print(test.columns)
print(test.sort_values('uncertain', ascending=False)[1:20])
