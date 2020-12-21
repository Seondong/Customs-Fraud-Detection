import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle

from collections import defaultdict
from itertools import islice, combinations
from datetime import datetime as dt
import warnings
warnings.filterwarnings("ignore")


def mask_labels(df: pd.DataFrame, ir_init: float, ssl_strategy: str) -> pd.DataFrame:
    """
    Masking certain amount of data for semi-supervised learning by specific strategy.
    ssl_strategy = importer
        Masking certain amount of importer_id, to mimic the situation that not all imports are inspected.
    ssl_strategy = = random
        Masking transactions by random sampling.
    ir_init is the inspection ratio at the beginning.
    """
    print('Before masking:\n', df['illicit'].value_counts())
    # To do: For more consistent results, we can control the random seed while selecting inspected_id.
    if ssl_strategy == "importer":
        inspected_id = {}
        train_id = list(set(df['importer.id']))
        inspected_id[ir_init] = np.random.choice(train_id, size= int(len(train_id) * ir_init / 100), replace=False)
        d = {}
        for id in train_id:
            d[id] = float('nan')
        for id in inspected_id[ir_init]:
            d[id] = 1
        df['illicit'] = df['importer.id'].apply(lambda x: d[x]) * df['illicit']
        df['revenue'] = df['importer.id'].apply(lambda x: d[x]) * df['revenue']
    elif ssl_strategy == "random":
        sampled_idx = list(df.sample(frac=1 - ir_init / 100, replace=False).index)
        df.loc[sampled_idx,"illicit"] = df.loc[sampled_idx,"illicit"]* np.nan
        df.loc[sampled_idx,"revenue"] = df.loc[sampled_idx,"revenue"]* np.nan

    print('After masking:\n', df['illicit'].value_counts())
    return df


def merge_attributes(df: pd.DataFrame, *args: str) -> None:
    """
    dtype df: dataframe
    dtype *args: strings (attribute names that want to be combined)
    """
    iterables = [df[arg].astype(str) for arg in args]
    columnName = '&'.join([*args]) 
    fs = [''.join([v for v in var]) for var in zip(*iterables)]
    df.loc[:, columnName] = fs
    
    
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    dtype df: dataframe
    rtype df: dataframe
    """
    if len(df) == 0:
        return df
    
    
    df = df.dropna(subset=['cif.value', 'total.taxes', 'quantity'])
    df.loc[:, 'Unitprice'] = df['cif.value']/df['quantity']
    df.loc[:, 'WUnitprice'] = df['cif.value']/df['gross.weight']
    df.loc[:, 'TaxRatio'] = df['total.taxes'] / df['cif.value']
    df.loc[:, 'TaxUnitquantity'] = df['total.taxes'] / df['quantity']
    df.loc[:, 'HS6'] = df['tariff.code'].apply(lambda x: int(x // 10000))
    df.loc[:, 'HS4'] = df['HS6'].apply(lambda x: int(x // 100))
    df.loc[:, 'HS2'] = df['HS4'].apply(lambda x: int(x // 100))

    
#     candFeaturesCombine = ['office.id','importer.id','country','HS6','declarant.id']
#     for subset in combinations(candFeaturesCombine, 2):
#         merge_attributes(df, *subset)
    
#     for subset in combinations(candFeaturesCombine, 3):
#         merge_attributes(df, *subset)
        
    merge_attributes(df, 'office.id','importer.id')
    merge_attributes(df, 'office.id','HS6')
    merge_attributes(df, 'office.id','country')
    merge_attributes(df, 'HS6','country')
    
    df['sgd.date'] = df['sgd.date'].apply(lambda x: dt.strptime(x, '%y-%m-%d'))
    df.loc[:, 'SGD.DayofYear'] = df['sgd.date'].dt.dayofyear
    df.loc[:, 'SGD.WeekofYear'] = df['sgd.date'].dt.weekofyear
    df.loc[:, 'SGD.MonthofYear'] = df['sgd.date'].dt.month
    return df


def find_risk_profile(df: pd.DataFrame, feature: str, topk_ratio: float, adj: float, option: str) -> list or dict:
    """
    dtype feature: str
    dtype topk_ratio: float (range: 0-1)
    dtype adj: float (to modify the mean)
    dtype option: str ('topk', 'ratio')
    rtype: list(option='topk') or dict(option='ratio')
    
    The option topk is usually better than the ratio because of overfitting.
    """

    # Top-k suspicious item flagging
    if option == 'topk':
        total_cnt = df.groupby([feature])['illicit']
        nrisky_profile = int(topk_ratio*len(total_cnt))+1
        # prob_illicit = total_cnt.mean()  # Simple mean
        adj_prob_illicit = total_cnt.sum() / (total_cnt.count()+adj)  # Smoothed mean
        return list(adj_prob_illicit.sort_values(ascending=False).head(nrisky_profile).index)
    
    # Illicit-ratio encoding (Mean target encoding)
    # Refer: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/target-encoding.html
    # Refer: https://towardsdatascience.com/why-you-should-try-mean-encoding-17057262cd0
    elif option == 'ratio':
        # For target encoding, we just use 70% of train data to avoid overfitting (otherwise, test AUC drops significantly)
        total_cnt = df.sample(frac=0.7).groupby([feature])['illicit']
        nrisky_profile = int(topk_ratio*len(total_cnt))+1
        # prob_illicit = total_cnt.mean()  # Simple mean
        adj_prob_illicit = total_cnt.sum() / (total_cnt.count()+adj)  # Smoothed mean
        return adj_prob_illicit.to_dict()
    
    
def tag_risky_profiles(df: pd.DataFrame, profile: str, profiles: list or dict, option: str) -> pd.DataFrame:
    """
    dtype df: dataframe
    dtype profile: str
    dtype profiles: list(option='topk') or dictionary(option='ratio')
    dtype option: str ('topk', 'ratio')
    rtype: dataframe
    
    The option topk is usually better than the ratio because of overfitting.
    """
    if len(df) == 0:
        return df
    
    # Top-k suspicious item flagging
    if option == 'topk':
        d = defaultdict(int)
        for id in profiles:
            d[id] = 1
    #     print(list(islice(d.items(), 10)))  # For debugging
        df.loc[:, 'RiskH.'+profile] = df[profile].apply(lambda x: d[x])
    
    # Illicit-ratio encoding
    elif option == 'ratio':
        overall_ratio_train = np.mean(train.illicit) # When scripting, saving it as a class variable is clearer.
        df.loc[:, 'RiskH.'+profile] = df[profile].apply(lambda x: profiles.get(x), overall_ratio_train)
    return df



class Import_declarations():
    """ Class for dataset engineering """
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path, encoding = "ISO-8859-1")
        self.profile_candidates = None
        self.firstCheck()
        
     
    def firstCheck(self):
        """ Sorting and indexing necessary for data preparation """
        self.df = self.df.dropna(subset=["illicit"])
        self.df = self.df.sort_values("sgd.date")
        self.df = self.df.reset_index(drop=True)
        

    def split(self, train_start_day, valid_start_day, test_start_day, test_end_day, valid_length, test_length, args):
        """ Split data into train / valid / test """
            
        self.train_start_day = train_start_day.strftime('%y-%m-%d')
        self.valid_start_day = valid_start_day.strftime('%y-%m-%d')
        self.test_start_day = test_start_day.strftime('%y-%m-%d')
        self.test_end_day = test_end_day.strftime('%y-%m-%d')
        self.valid_length = valid_length
        self.test_length = test_length
        self.args = args
        
        self.train = self.df[(self.df["sgd.date"] >= self.train_start_day) & (self.df["sgd.date"] < self.valid_start_day)]
        self.valid = self.df[(self.df["sgd.date"] >= self.valid_start_day) & (self.df["sgd.date"] < self.test_start_day)]    
        self.test = self.df[(self.df["sgd.date"] >= self.test_start_day) & (self.df["sgd.date"] < self.test_end_day)]  
        
        # Intentionally masking datasets to simulate partially labeled scenario, note that our dataset is 100% inspected.
        # If your dataset is partially labeled already, comment below two lines.
        if args.data in ['synthetic', 'real-n', 'real-m', 'real-t']:
            self.train = mask_labels(self.train, args.initial_inspection_rate, args.ssl_strategy)
      
        self.train_lab = self.train[self.train['illicit'].notna()]
        self.train_unlab = self.train[self.train['illicit'].isna()]
        self.valid_lab = self.valid[self.valid['illicit'].notna()]
        self.valid_unlab = self.valid[self.valid['illicit'].isna()]
      
        # save labels
        self.train_cls_label = self.train_lab["illicit"].values
        self.valid_cls_label = self.valid_lab["illicit"].values
        self.test_cls_label = self.test["illicit"].values
        self.train_reg_label = self.train_lab['revenue'].values
        self.valid_reg_label = self.valid_lab['revenue'].values
        self.test_reg_label = self.test['revenue'].values
        
        # Normalize revenue labels for later model fitting
        self.norm_revenue_train, self.norm_revenue_valid, self.norm_revenue_test = np.log(self.train_reg_label+1), np.log(self.valid_reg_label+1), np.log(self.test_reg_label+1) 
        global_max = max(self.norm_revenue_train) 
        self.norm_revenue_train = self.norm_revenue_train/global_max
        self.norm_revenue_valid = self.norm_revenue_valid/global_max
        self.norm_revenue_test = self.norm_revenue_test/global_max
        
        self.train_valid_lab = pd.concat([self.train_lab, self.valid_lab])
        self.train_valid_unlab = pd.concat([self.train_unlab, self.valid_unlab])
        
        
    def featureEngineering(self):
        """ Feature engineering, """
        self.semi_supervised = self.args.semi_supervised
        self.offset = self.test.index[0]

        # Run preprocessing
        self.train_lab = preprocess(self.train_lab)
        self.train_unlab = preprocess(self.train_unlab)
        self.valid_lab = preprocess(self.valid_lab)
        self.valid_unlab = preprocess(self.valid_unlab)
        self.test = preprocess(self.test)
        
        
        # Add a few more risky profiles
        risk_profiles = {}
        profile_candidates = self.profile_candidates + [col for col in self.train_lab.columns if '&' in col]

        for profile in profile_candidates:
            option = 'topk'
            risk_profiles[profile] = find_risk_profile(self.train_lab, profile, 0.1, 10, option=option)
            self.train_lab = tag_risky_profiles(self.train_lab, profile, risk_profiles[profile], option=option)
            self.train_unlab = tag_risky_profiles(self.train_unlab, profile, risk_profiles[profile], option=option)
            self.valid_lab = tag_risky_profiles(self.valid_lab, profile, risk_profiles[profile], option=option)
            self.valid_unlab = tag_risky_profiles(self.valid_unlab, profile, risk_profiles[profile], option=option)
            self.test = tag_risky_profiles(self.test, profile, risk_profiles[profile], option=option)
        
        # Features to use in a classifier
        self.column_to_use = ['cif.value', 'total.taxes', 'gross.weight', 'quantity', 'Unitprice', 'WUnitprice', 'TaxRatio', 'TaxUnitquantity', 'tariff.code', 'HS6', 'HS4', 'HS2', 'SGD.DayofYear', 'SGD.WeekofYear', 'SGD.MonthofYear'] + [col for col in self.train_lab.columns if 'RiskH' in col] 
        
        self.X_train_lab = self.train_lab[self.column_to_use].values
        if not self.train_unlab.empty:
            self.X_train_unlab = self.train_unlab[self.column_to_use].values
        else:
            self.X_train_unlab = np.asarray([])
        self.X_valid_lab = self.valid_lab[self.column_to_use].values
        if not self.valid_unlab.empty:
            self.X_valid_unlab = self.valid_unlab[self.column_to_use].values
        else:
            self.X_valid_unlab = np.asarray([])
        self.X_test = self.test[self.column_to_use].values
        print("Data size:")
        print(f'Train labeled: {self.train_lab.shape}, Train unlabeled: {self.train_unlab.shape}, Valid labeled: {self.valid_lab.shape}, Valid unlabeled: {self.valid_unlab.shape}, Test: {self.test.shape}')

        # impute nan
        self.X_train_lab = np.nan_to_num(self.X_train_lab, 0)
        self.X_train_unlab = np.nan_to_num(self.X_train_unlab, 0)
        self.X_valid_lab = np.nan_to_num(self.X_valid_lab, 0)
        self.X_valid_unlab = np.nan_to_num(self.X_valid_unlab, 0)
        self.X_test = np.nan_to_num(self.X_test, 0)

        from collections import Counter
        print("Checking label distribution")
        cnt = Counter(self.train_cls_label)
        print("Training:",cnt[1]/cnt[0])
        cnt = Counter(self.valid_cls_label)
        print("Validation:",cnt[1]/cnt[0])
        cnt = Counter(self.test_cls_label)
        print("Testing:",cnt[1]/cnt[0])
        
        self.dftrainx_lab = pd.DataFrame(self.X_train_lab,columns=self.column_to_use)
        try:
            self.dftrainx_unlab = pd.DataFrame(self.X_train_unlab,columns=self.column_to_use)
        except:
            self.dftrainx_unlab = pd.DataFrame(columns=self.column_to_use)
        self.dfvalidx_lab = pd.DataFrame(self.X_valid_lab,columns=self.column_to_use) 
        try:
            self.dfvalidx_unlab = pd.DataFrame(self.X_valid_unlab,columns=self.column_to_use)
        except:
            self.dfvalidx_unlab = pd.DataFrame(columns=self.column_to_use)
        self.dftestx = pd.DataFrame(self.X_test,columns=self.column_to_use)
    
    
    def update(self, inspected_imports, uninspected_imports, test_start_day, test_end_day, valid_start_day):
        """ Update the dataset for next test phase. 
            Newly inspected imports are updated to train-labeled data, newly uninspected imports are updated to train-unlabeled data. """
        
        
        self.train_valid_lab = pd.concat([self.train_valid_lab, inspected_imports]).sort_index()
        self.train_valid_unlab = pd.concat([self.train_valid_unlab, uninspected_imports]).sort_index()
        
        self.test_start_day = test_start_day.strftime('%y-%m-%d')
        self.test_end_day = test_end_day.strftime('%y-%m-%d')
        self.valid_start_day = valid_start_day.strftime('%y-%m-%d')
             
        self.train_lab = self.train_valid_lab[self.train_valid_lab["sgd.date"] < self.valid_start_day]
        self.valid_lab = self.train_valid_lab[self.train_valid_lab["sgd.date"] >= self.valid_start_day]
        self.train_unlab = self.train_valid_unlab[self.train_valid_unlab["sgd.date"] < self.valid_start_day]
        self.valid_unlab = self.train_valid_unlab[self.train_valid_unlab["sgd.date"] >= self.valid_start_day]
        
        # From here, just updating relevant items 
        self.train = pd.concat([self.train_lab, self.train_unlab]).sort_index()
        self.valid = pd.concat([self.valid_lab, self.valid_unlab]).sort_index()
        self.test = self.df[(self.df["sgd.date"] >= self.test_start_day) & (self.df["sgd.date"] < self.test_end_day)]  
        
        self.train_cls_label = self.train_lab["illicit"].values
        self.valid_cls_label = self.valid_lab["illicit"].values
        self.test_cls_label = self.test["illicit"].values
        self.train_reg_label = self.train_lab['revenue'].values
        self.valid_reg_label = self.valid_lab['revenue'].values
        self.test_reg_label = self.test['revenue'].values
        
        self.norm_revenue_train, self.norm_revenue_valid, self.norm_revenue_test = np.log(self.train_reg_label+1), np.log(self.valid_reg_label+1), np.log(self.test_reg_label+1) 
        global_max = max(self.norm_revenue_train) 
        self.norm_revenue_train = self.norm_revenue_train/global_max
        self.norm_revenue_valid = self.norm_revenue_valid/global_max
        self.norm_revenue_test = self.norm_revenue_test/global_max
        
        
        
class Syntheticdata(Import_declarations):
    """ Class for synthetic data
    
    ToDo: For later usage, we should support flexibility towards different datasets with different features. End-users would like to use their existing features. For this program, initial preprocessings were done to make columns consistently.
    """
    def __init__(self, path):
        super(Syntheticdata, self).__init__(path)
        self.profile_candidates = ['importer.id', 'declarant.id', 'tariff.code', 'quantity', 'HS6', 'HS4', 'HS2', 'office.id']
        
        
        
class Ndata(Import_declarations):
    """ Class for Ndata"""
    def __init__(self, path):
        super(Ndata, self).__init__(path)
        self.profile_candidates = ['importer.id', 'declarant.id', 'tariff.code', 'quantity', 'HS6', 'HS4', 'HS2', 'office.id']

        
class Mdata(Import_declarations):
    """ Class for Mdata"""
    def __init__(self, path):
        super(Mdata, self).__init__(path)
        self.profile_candidates = ['importer.id', 'exporter.name', 'expcty', 'country', 'declarant.id', 'tariff.code', 'quantity', 'HS6', 'HS4', 'HS2', 'office.id']
        
        
class Tdata(Import_declarations):
    """ Class for Tdata"""
    def __init__(self, path):
        super(Tdata, self).__init__(path)
        self.profile_candidates = ['importer.id', 'country', 'last.departure.code', 'contract.party.code',
                      'tariff.code', 'quantity', 'HS6', 'HS4', 'HS2', 'office.id']


class Cdata(Import_declarations):
    """ Class for Cdata - waiting"""
    def __init__(self, path):
        super(Cdata, self).__init__(path)


class Kdata(Import_declarations):
    """ Class for Kdata - waiting"""
    def __init__(self, path):
        super(Kdata, self).__init__(path)

        




        
    

