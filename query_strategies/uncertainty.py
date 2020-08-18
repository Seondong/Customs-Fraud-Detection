import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

class Uncertainty :
    numerical_features = ['fob.value', 'cif.value', 'total.taxes', 'gross.weight', 'quantity', 'Unitprice', 'WUnitprice', 'TaxRatio', 'FOBCIFRatio', 'TaxUnitquantity']
    categorical_features = ['RiskH.importer.id', 'RiskH.declarant.id',
        'RiskH.HS6.Origin', 'RiskH.tariff.code', 'RiskH.HS6',
        'RiskH.HS4', 'RiskH.HS2', 'RiskH.office.id']
    column_to_use_unc_measure = numerical_features + categorical_features

    def __init__(self, labeled_data) :
        self.classifiers = dict()
        self.regressors = dict()
        self.data = pd.DataFrame(labeled_data, columns = self.column_to_use_unc_measure)
        self.importance_classifier = None

    def train(self) :
        for cc in self.categorical_features :
            columns = [col for col in self.column_to_use_unc_measure if col != cc]
            train_set = pd.DataFrame(self.data, columns = columns)
            xgb_clf = XGBClassifier(n_jobs=-1)
            xgb_clf.fit(train_set ,self.data[cc].values)
            self.classifiers[cc] = xgb_clf
        
        for nc in self.numerical_features :
            columns = [col for col in self.column_to_use_unc_measure if col != nc]
            train_set = pd.DataFrame(self.data, columns = columns)
            xgb_reg = XGBRegressor(n_jobs=-1)
            xgb_reg.fit(train_set, self.data[nc].values)
            self.regressors[nc] = xgb_reg

    def measure(self, test_data, option) :
        unc = pd.DataFrame()

        for cc in self.categorical_features :
            columns = [col for col in self.column_to_use_unc_measure if col != cc]
            test_set = pd.DataFrame(test_data, columns = columns)
            xgb_clf_pred = self.classifiers[cc].predict(test_set)
            unc['unc.'+cc] = np.bitwise_xor(test_data[cc], xgb_clf_pred.tolist())
        
            for idx, cat in enumerate(test_data[cc[6:]]) :
                if cat not in set(self.data[cc[6:]]) :
                    unc['unc.'+cc][idx] = 1
            
        for nc in self.numerical_features :
            columns = [col for col in self.column_to_use_unc_measure if col != nc]
            test_set = pd.DataFrame(test_data, columns = columns)
            xgb_reg_pred = self.regressors[nc].predict(test_set)
            unc['unc.'+nc] = abs(test_data[nc] - xgb_reg_pred.tolist()) / test_data[nc]
            unc['unc.'+nc] = np.clip(np.asarray(unc['unc.'+nc]), 0, 1)

        if option == 'naive' :
            # Model 1 : Naive equally-contributing uncertainty (mean)
            return unc.mean(axis=1)

        elif option == 'feature_importance' :
            # Model 2 : Feature importance from illicitness
            self.importance_classifier = XGBClassifier(n_estimators = 100)
            self.importance_classifier.fit(pd.DataFrame(self.data, columns=self.column_to_use_unc_measure), pd.DataFrame(self.data, columns=['illicit']).values.ravel())
            return unc.dot(self.importance_classifier.feature_importances_ / sum(self.importance_classifier.feature_importances_))
        
    def retrain(self, queried_samples) :
        for cc in self.categorical_features :
            columns = [col for col in self.column_to_use_unc_measure if col != cc]
            train_set = pd.DataFrame(queried_samples, columns = columns)
            self.classifiers[cc].fit(train_set, queried_samples[cc].values, xgb_model = self.classifiers[cc])
        
        for nc in self.numerical_features :
            columns = [col for col in self.column_to_use_unc_measure if col != nc]
            train_set = pd.DataFrame(queried_samples, columns = columns)
            self.regressors[nc].fit(train_set, queried_samples[nc].values, xgb_model = self.regressors[nc])
        
        self.importance_classifier.fit(pd.DataFrame(queried_samples, columns = self.column_to_use_unc_measure), pd.DataFrame(queried_samples, columns=['illicit']).values.ravel(), xgb_model = self.importance_classifier)
        self.data.append(pd.DataFrame(queried_samples, columns = self.column_to_use_unc_measure))