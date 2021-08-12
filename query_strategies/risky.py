import numpy as np
import random
from .strategy import Strategy
from utils import timer_func
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader
import torch
import copy
import math
from tqdm import tqdm
import pandas as pd

class RiskProfileSampling(Strategy):
    """ Naive Risk Profile Sampling strategy (sum) """
    

    def __init__(self, args):
        super(RiskProfileSampling, self).__init__(args)


    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        self.y_totalrisk = self.data.dftestx[[col for col in self.data.dftestx.columns if 'RiskH' in col]].sum(axis=1)
        
    @timer_func
    def query(self, k):
        self.predict_frauds()
        chosen = np.argpartition(self.y_totalrisk[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()
    

    
class RiskProfileProdSampling(Strategy):
    """ Naive Risk Profile Sampling strategy (prod) """
    

    def __init__(self, args):
        super(RiskProfileProdSampling, self).__init__(args)


    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        self.y_totalrisk = self.data.dftestx[[col for col in self.data.dftestx.columns if ('RiskH' in col and '&' not in col)]].prod(axis=1)
        
    @timer_func
    def query(self, k):
        self.predict_frauds()
        chosen = np.argpartition(self.y_totalrisk[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()
    
    
    
class RiskProfileLogisticSampling(Strategy):
    """ Risk Profile with logistic regression Sampling strategy """
    

    def __init__(self, args):
        super(RiskProfileLogisticSampling, self).__init__(args)


    def train_weights(self):
        X = self.data.dftrainx_lab[[col for col in self.data.dftrainx_lab.columns if 'RiskH' in col]]
        y = self.data.train_cls_label
        self.clf = LogisticRegression(random_state=0).fit(X, y)
        
        
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        testX = self.data.dftestx[[col for col in self.data.dftestx.columns if 'RiskH' in col]]
        self.y_totalrisk = self.clf.predict_proba(testX)[:, 1]
        
        
    @timer_func
    def query(self, k):
        self.train_weights()
        self.predict_frauds()
        chosen = np.argpartition(self.y_totalrisk[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()
    
    
class RiskProfilePrecisionSampling(Strategy):
    """ Risk Profile with Precision Sampling strategy """
    """ https://arxiv.org/pdf/1505.06813.pdf """
    

    def __init__(self, args):
        super(RiskProfilePrecisionSampling, self).__init__(args)
        self.start_flag = True


    def train_weights(self):
        self.trainX = self.data.dftrainx_lab[[col for col in self.data.train_lab.columns if 'RiskH' in col]].fillna(0)
        self.trainy = self.data.train_cls_label
        self.validX = self.data.dfvalidx_lab[[col for col in self.data.train_lab.columns if 'RiskH' in col]].fillna(0)
        self.validy = self.data.valid_cls_label
        self.train_ds = TensorDataset(torch.tensor(self.trainX.to_numpy()),torch.tensor(self.trainy))
        self.train_loader = DataLoader(self.train_ds, batch_size=min(len(self.train_ds) // 10, 256), shuffle=True)
        
        epochs = 10
        self.W = torch.zeros(self.trainX.shape[1], 1).double()
        best_precision = 0
        lr = 0.01
        for e in tqdm(range(epochs)):
            for i, batch in enumerate(self.train_loader):
                x, y = batch
                s_t = torch.matmul(x, self.W)
                yhat_t = torch.zeros_like(y).scatter_(0, s_t.topk(k = s_t.shape[0] // 10, dim=0).indices.squeeze(), 1)
                delta_t = ((1-y) * yhat_t).sum()
                if delta_t == 0:
                    continue
                else:
                    K = (y * yhat_t).sum()
                    Denom = ((y * y).sum() - K)
                    if Denom == 0:
                        continue
                    
                    D_t = delta_t / Denom
                    self.W = self.W - lr * ((1-y)*yhat_t * x.t()).sum(dim=1).unsqueeze(1)
                    self.W = self.W + lr * (D_t * ((1-yhat_t)*y * x.t()).sum(dim=1).unsqueeze(1))

            self.data.valid['risk_perceptron'] = torch.matmul(torch.tensor(self.validX.to_numpy()), self.W).detach().numpy()
            validsorted = self.data.valid[['illicit', 'risk_perceptron']].sort_values('risk_perceptron', ascending=False)
            length_valid = len(validsorted)
            for i in [0.01, 0.02, 0.05, 0.1]:
                validhead = validsorted.head(int(length_valid*i ))
                precision = validhead['illicit'].sum() / len(validhead)
                recall = validhead['illicit'].sum() / validsorted['illicit'].sum()
                print(precision, recall, len(validhead))

            if precision > best_precision:
                self.bestW = copy.copy(self.W)
                best_precision = precision
        
        
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        print(self.bestW.squeeze())
        testX = self.data.dftestx[[col for col in self.data.dftestx.columns if 'RiskH' in col]]
        self.y_totalrisk = torch.matmul(torch.tensor(testX.to_numpy()), self.bestW).squeeze().detach().numpy()
        
        
    @timer_func
    def query(self, k):
        self.train_weights()
        self.predict_frauds()
        chosen = np.argpartition(self.y_totalrisk[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()
    
    
class RiskProfileMABSampling(Strategy):
    """ Risk Profile with Multi Armed Bandit Sampling strategy """
    """ Thompson's sampling will be utilized (not discounted) """
    """ Product of beta sampled value is used """
    
    def __init__(self, args):
        super(RiskProfileMABSampling, self).__init__(args)

        
    def calculate_beta_parameter(self):
        category_list = self.data.profile_candidates
        self.cat_dict_dict = {}
        for category in category_list:
            cat_df = self.data.train_lab.groupby(category)['illicit'].agg(['sum','count'])
            cat_df['alpha'] = cat_df['sum'] + 1
            cat_df['beta'] = cat_df['count'] + 1 - cat_df['sum']   
            cat_dict = cat_df[['alpha', 'beta']].to_dict()
            self.cat_dict_dict[category] = cat_dict

            
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        test_df = self.data.test[self.data.profile_candidates]
        for profile in tqdm(self.data.profile_candidates):
            current_dict = self.cat_dict_dict[profile]
            test_df[profile +'-sample'] = test_df[profile].apply(lambda x: random.betavariate(current_dict['alpha'].get(x, 1), 
                                                                                              current_dict['beta'].get(x, 1)))
        self.y_totalrisk = test_df[[col for col in test_df.columns if '-sample' in col]].prod(axis=1).to_numpy().squeeze()
        
        
    @timer_func
    def query(self, k):
        self.calculate_beta_parameter()
        self.predict_frauds()
        chosen = np.argpartition(self.y_totalrisk[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()
    
    
    
class RiskProfileMABSumSampling(Strategy):
    """ Risk Profile with Multi Armed Bandit Sampling strategy """
    """ Thompson's sampling will be utilized (not discounted) """
    """ Sum of beta sampled value is used """
    
    def __init__(self, args):
        super(RiskProfileMABSumSampling, self).__init__(args)

        
    def calculate_beta_parameter(self):
        category_list = self.data.profile_candidates
        self.cat_dict_dict = {}
        for category in category_list:
            cat_df = self.data.train_lab.groupby(category)['illicit'].agg(['sum','count'])
            cat_df['alpha'] = cat_df['sum'] + 1
            cat_df['beta'] = cat_df['count'] + 1 - cat_df['sum']   
            cat_dict = cat_df[['alpha', 'beta']].to_dict()
            self.cat_dict_dict[category] = cat_dict

            
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        test_df = self.data.test[self.data.profile_candidates]
        for profile in tqdm(self.data.profile_candidates):
            current_dict = self.cat_dict_dict[profile]
            test_df[profile +'-sample'] = test_df[profile].apply(lambda x: random.betavariate(current_dict['alpha'].get(x, 1), 
                                                                                              current_dict['beta'].get(x, 1)))
        self.y_totalrisk = test_df[[col for col in test_df.columns if '-sample' in col]].sum(axis=1).to_numpy().squeeze()
        
        
    @timer_func
    def query(self, k):
        self.calculate_beta_parameter()
        self.predict_frauds()
        chosen = np.argpartition(self.y_totalrisk[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()
    
    
    
class RiskProfileDiscountMABSampling(Strategy):
    """ Risk Profile with Multi Armed Bandit Sampling strategy """
    """ Discounted Thompson's sampling will be utilized  """
    """ Product of beta sampled value is used """
    
    def __init__(self, args):
        super(RiskProfileDiscountMABSampling, self).__init__(args)
        self.decay = math.pow(0.9, (1/90))

        
    def calculate_beta_parameter(self):
        category_list = self.data.profile_candidates
        self.data.train_lab['sgd.date'] = pd.to_datetime(self.data.train_lab['sgd.date'], format='%y-%m-%d')
        end_time = self.data.train_lab['sgd.date'].iloc[-1]
        self.data.train_lab['gamma'] = self.data.train_lab['sgd.date'].apply(lambda x: math.pow(self.decay, (end_time - x).days))
        self.data.train_lab['decayed_illicit'] = self.data.train_lab['illicit'] * self.data.train_lab['gamma']
        self.data.train_lab['decayed_nonillicit'] = (1 - self.data.train_lab['illicit']) * self.data.train_lab['gamma']
        
        self.cat_dict_dict = {}
        for category in category_list:
            cat_df = self.data.train_lab.groupby(category)[['decayed_illicit', 'decayed_nonillicit']].sum()
            cat_df['alpha'] = cat_df['decayed_illicit'] + 1
            cat_df['beta'] = cat_df['decayed_nonillicit'] + 1
            cat_dict = cat_df[['alpha', 'beta']].to_dict()
            self.cat_dict_dict[category] = cat_dict

            
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        test_df = self.data.test[self.data.profile_candidates]
        for profile in tqdm(self.data.profile_candidates):
            current_dict = self.cat_dict_dict[profile]
            test_df[profile +'-sample'] = test_df[profile].apply(lambda x: random.betavariate(current_dict['alpha'].get(x, 1), 
                                                                                              current_dict['beta'].get(x, 1)))
        self.y_totalrisk = test_df[[col for col in test_df.columns if '-sample' in col]].prod(axis=1).to_numpy().squeeze()
        
        
    @timer_func
    def query(self, k):
        self.calculate_beta_parameter()
        self.predict_frauds()
        chosen = np.argpartition(self.y_totalrisk[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()
    
    
class RiskProfileDiscountMABSumSampling(Strategy):
    """ Risk Profile with Multi Armed Bandit Sampling strategy """
    """ Discounted Thompson's sampling will be utilized  """
    """ Sum of beta sampled value is used """
    
    def __init__(self, args):
        super(RiskProfileDiscountMABSumSampling, self).__init__(args)
        self.decay = math.pow(0.9, (1/90))

        
    def calculate_beta_parameter(self):
        category_list = self.data.profile_candidates
        self.data.train_lab['sgd.date'] = pd.to_datetime(self.data.train_lab['sgd.date'], format='%y-%m-%d')
        end_time = self.data.train_lab['sgd.date'].iloc[-1]
        self.data.train_lab['gamma'] = self.data.train_lab['sgd.date'].apply(lambda x: math.pow(self.decay, (end_time - x).days))
        self.data.train_lab['decayed_illicit'] = self.data.train_lab['illicit'] * self.data.train_lab['gamma']
        self.data.train_lab['decayed_nonillicit'] = (1 - self.data.train_lab['illicit']) * self.data.train_lab['gamma']
        
        self.cat_dict_dict = {}
        for category in category_list:
            cat_df = self.data.train_lab.groupby(category)[['decayed_illicit', 'decayed_nonillicit']].sum()
            cat_df['alpha'] = cat_df['decayed_illicit'] + 1
            cat_df['beta'] = cat_df['decayed_nonillicit'] + 1
            cat_dict = cat_df[['alpha', 'beta']].to_dict()
            self.cat_dict_dict[category] = cat_dict

            
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        test_df = self.data.test[self.data.profile_candidates]
        for profile in tqdm(self.data.profile_candidates):
            current_dict = self.cat_dict_dict[profile]
            test_df[profile +'-sample'] = test_df[profile].apply(lambda x: random.betavariate(current_dict['alpha'].get(x, 1), 
                                                                                              current_dict['beta'].get(x, 1)))
        self.y_totalrisk = test_df[[col for col in test_df.columns if '-sample' in col]].sum(axis=1).to_numpy().squeeze()
        
        
    @timer_func
    def query(self, k):
        self.calculate_beta_parameter()
        self.predict_frauds()
        chosen = np.argpartition(self.y_totalrisk[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()