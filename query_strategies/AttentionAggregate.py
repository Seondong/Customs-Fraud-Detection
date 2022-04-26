import pandas as pd
import numpy as np
import random
import sys
import pickle
import sys
import math
import gc
import os
import warnings
sys.path.append("..")
warnings.filterwarnings("ignore")

from .strategy import Strategy

import time
import copy
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from torch.utils import data
from torchtools.optim import RangerLars

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from utils import timer_func


class AttentionSampling(Strategy):
    """ Attention based model """
    
    def __init__(self, args):
        super(AttentionSampling,self).__init__(args)
        self.neighbor_k = 50
        self.hidden_size = 32
        self.batch_size = args.batch_size
        self.epoch = args.epoch

        
    def generate_metagraph(self):
        self.xgb = XGBClassifier(n_estimators=5, max_depth=7, n_jobs=-1, eval_metric='logloss', verbosity = 0)
        self.xgb.fit(self.data.dftrainx_lab.to_numpy(), self.data.train_cls_label) 
        
        X_train_leaves = self.xgb.apply(self.data.dftrainx_lab.to_numpy())
        X_train_unlab_leaves = self.xgb.apply(self.data.dftrainx_unlab.to_numpy())
        X_valid_leaves = self.xgb.apply(self.data.dfvalidx_lab.to_numpy())
        X_test_leaves = self.xgb.apply(self.data.dftestx.to_numpy())
        
        self.data.train_lab[[f'tree{i}' for i in range(5)]] = X_train_leaves
        self.data.train_unlab[[f'tree{i}' for i in range(5)]] = X_train_unlab_leaves
        self.data.valid_lab[[f'tree{i}' for i in range(5)]] = X_valid_leaves
        self.data.test[[f'tree{i}' for i in range(5)]] = X_test_leaves 
        self.category = [f'tree{i}' for i in range(5)]
        
        
    def prepare_dataloader(self):
        self.train_ds = AttDataset(self.data, self.data.train_lab, self.data.train_unlab, self.data.train_lab, self.neighbor_k, self.category)
        self.valid_ds = AttDataset(self.data, self.data.train_lab, self.data.train_unlab, self.data.valid_lab, self.neighbor_k, self.category)
        train_valid_all_df = pd.concat([self.data.train_lab, self.data.valid_lab])
        self.test_ds = AttDataset(self.data, train_valid_all_df, self.data.train_unlab, self.data.test, self.neighbor_k, self.category)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=8, shuffle=True, drop_last=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=False)
        
        
    def train_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = len(self.data.column_to_use)+1
        self.model = AttDetect(self.input_dim, self.hidden_size, self.category)
        self.model.to(device)
        
        optimizer = RangerLars(self.model.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.8)
        
        best_f1_top = 0
        for epoch in range(self.epoch):
            print("epoch: ", epoch)
            self.model.train()
            loss_avg = 0
            for i, batch in enumerate(tqdm(self.train_loader)):
                row_feature, neighbor_stack, row_target = batch
                row_feature = row_feature.to(device)
                neighbor_stack = neighbor_stack.to(device)
                row_target = row_target.to(device)
                loss, logits = self.model(row_feature, neighbor_stack, row_target)
                loss_avg += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            print('train_loss: ', loss_avg / len(self.train_loader.dataset))
                
            # validation eval
            self.model.eval()
            with torch.no_grad():
                logit_list = []
                for i, batch in enumerate(tqdm(self.valid_loader)):
                    row_feature, neighbor_stack, row_target = batch
                    row_feature = row_feature.to(device)
                    neighbor_stack = neighbor_stack.to(device)
                    row_target = row_target.to(device)     
                    loss, logits = self.model(row_feature, neighbor_stack, row_target)
                    logit_list.append(logits.reshape(-1, 1))
                
                outputs = torch.cat(logit_list).detach().cpu().numpy().ravel()
                f, pr, re = torch_metrics(np.array(outputs), self.valid_ds.target_Y.astype(int).to_numpy())
                f1_top = np.mean(f)
                
            if f1_top > best_f1_top:
                self.best_model = self.model
                best_f1_top = f1_top
                
    
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model.eval()
        logit_list = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                row_feature, neighbor_stack, _ = batch
                row_feature = row_feature.to(device)
                neighbor_stack = neighbor_stack.to(device)
                logits = self.model(row_feature, neighbor_stack)
                logit_list.append(logits.reshape(-1, 1))        

        self.y_prob = torch.cat(logit_list).detach().cpu().numpy().ravel()     
        
    @timer_func
    def query(self, k):
        if self.args.semi_supervised == 0:
            sys.exit('(AttentionAgg is a semi-supervised algorithm, check if the parameter --semi_supervised is set as 1')
        self.generate_metagraph()
        self.prepare_dataloader()
        self.train_model()
        self.predict_frauds()
        chosen = np.argpartition(self.y_prob[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()
    

def torch_metrics(y_prob, xgb_testy, display=True):
    """ Evaluate the performance"""
    pr, re, f = [], [], []
    # For validatation, we measure the performance on 5% (previously, 1%, 2%, 5%, and 10%)
    for i in [99,98,95,90]: 
        threshold = np.percentile(y_prob, i)
        precision = xgb_testy[y_prob >= threshold].mean()    # y_prob이 1인 경우가 많아서 > 를 >=로 바꿈. 아니면 nan이 나와서 f1-top계산이 안됨.
        recall = sum(xgb_testy[y_prob >= threshold])/ sum(xgb_testy)
        f1 = 2 * (precision * recall) / (precision + recall)
        if display:
            print(f'Checking top {100-i}% suspicious transactions: {len(y_prob[y_prob > threshold])}')
            print('Precision: %.4f, Recall: %.4f' % (precision, recall))
        # save results
        pr.append(precision)
        re.append(recall)
        f.append(f1)
    return f, pr, re    
    
    
    
class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x    
   
    
class AttDataset(torch.utils.data.Dataset):
    def __init__(self, data, labeled_search_df, unlabeled_search_df, target_df, neighbor_k, category):
        self.neighbor_k = neighbor_k
        self.target_df = target_df.fillna(0)   
        self.data = data

        self.search_df = pd.concat([labeled_search_df, unlabeled_search_df]).reset_index().fillna(0)        
        self.labeled_search_df = self.search_df[self.search_df['illicit']!=0.5].fillna(0)   
        self.unlabeled_search_df = self.search_df[self.search_df['illicit']==0.5].fillna(0)   
        
        self.idx_dict = self.labeled_search_df['index'].T.to_dict()  
        self.target_Y = self.target_df['illicit'].copy()
        
        self.category = category
        self.group_dict = {}  
        self.node_columns = data.column_to_use+['illicit']

        self.labeled_nested_dict = {}
        for cat in self.category:
            cat_dict = {}
            grouped_df = self.labeled_search_df.groupby(cat)
            for group_name, group in grouped_df:
                cat_dict[group_name] = group.index.to_numpy()
            self.labeled_nested_dict[cat] = cat_dict
            
        self.unlabeled_nested_dict = {}
        for cat in self.category:
            cat_dict = {}
            grouped_df = self.unlabeled_search_df.groupby(cat)
            for group_name, group in grouped_df:
                cat_dict[group_name] = group.index.to_numpy()
            self.unlabeled_nested_dict[cat] = cat_dict

    def __len__(self):
        return len(self.target_df)
    
    def __getitem__(self, idx, train=False):
        row = self.target_df.iloc[idx]
        row_feature = torch.tensor(row[self.node_columns].to_numpy(dtype=float)).float()
        row_feature[-1] = 0.5
        row_target = int(self.target_Y.iloc[idx])
        
        neighbor_list = []
        for cat in self.category:
            labeled_cat_dict = self.labeled_nested_dict[cat]
            unlabeled_cat_dict = self.unlabeled_nested_dict[cat]     
            
            labeled_index = labeled_cat_dict.get(row[cat], np.array([])).copy()
            unlabeled_index = unlabeled_cat_dict.get(row[cat], np.array([]))
            
            if len(labeled_index) + len(unlabeled_index) > 0:
                if len(labeled_index) < self.neighbor_k:
                    labeled_neighbors = labeled_index
                    sample_num = min(self.neighbor_k - len(labeled_neighbors), len(unlabeled_index))
                    random_idx = (torch.rand(sample_num*2) * len(unlabeled_index)).numpy().astype(int)
                    unlabeled_neighbors = list(dict.fromkeys(unlabeled_index[random_idx]))[:sample_num]
                    neighbors = np.concatenate((labeled_neighbors, unlabeled_neighbors), axis=0)
                else:
                    sample_num = self.neighbor_k
                    random_idx = torch.rand(sample_num*2)
                    random_idx = random_idx * len(labeled_index)
                    random_idx = random_idx.numpy().astype(int)
                    neighbors = list(dict.fromkeys(labeled_index[random_idx]))[:sample_num]
     
                aggregated_neighbors = torch.tensor(self.search_df.iloc[neighbors][self.node_columns].to_numpy()).float()                
                if len(aggregated_neighbors) < self.neighbor_k:
                    diff = self.neighbor_k - len(aggregated_neighbors)
                    aggregated_neighbors = torch.cat((aggregated_neighbors, torch.zeros(diff, aggregated_neighbors.shape[1])))
            else:
                aggregated_neighbors = torch.zeros(self.neighbor_k, row_feature.shape[0])
            neighbor_list.append(aggregated_neighbors)

        neighbor_stack = torch.stack(neighbor_list, dim=0)
        return row_feature, neighbor_stack, row_target


    
class AttDetect(nn.Module):
    def __init__(self, input_dim, hidden_dim, category, nhead=8):
        super(AttDetect, self).__init__()    
        self.input_dim = input_dim
        self.dim = hidden_dim
        self.nhead = nhead
        self.category = category
        
        self.initEmbedding = MLP(self.input_dim, self.dim, self.dim)
        self.selfmha_list = nn.ModuleList()
        self.mha_list = nn.ModuleList()
        for cat in category:
            self.selfmha_list.append(nn.MultiheadAttention(self.dim, self.nhead, dropout=0.1))
            self.mha_list.append(nn.MultiheadAttention(self.dim, self.nhead, dropout=0.1))

        self.finall_addatt = AdditiveAttention(self.dim, self.dim)
        self.classifier = nn.Linear(self.dim, 1)
        
        self.recons_layer = nn.Sequential(nn.Linear(self.dim, self.dim),
                                          nn.BatchNorm1d(self.dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.dim, self.input_dim))
        
        self.recon_criterion = nn.MSELoss()
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, row_feature, neighbor_stack, labels=None, recon_target=None, mask=None, attention=False):
        batch_sz, n_kind, n_neighbor, feature_dim = neighbor_stack.shape # [B, K, N, D]
        neighbor_stack = neighbor_stack.reshape(-1, feature_dim)
        
        query_embed = self.initEmbedding(row_feature) 
        neighbor_embed = self.initEmbedding(neighbor_stack)
        neighbor_embed = neighbor_embed.reshape(batch_sz, n_kind, n_neighbor, -1)
        neighbor_embed = neighbor_embed.permute(1, 0, 2, 3) # [K, B, N, D]
        
        neighbor_queryatt_list = []
        neighbor_attvec_list = []
        for i, neighbor_each in enumerate(neighbor_embed):
            neighbor_each = F.dropout(neighbor_each.permute(1, 0, 2), 0.1) # [N, B, D]
            neighbor_coatt, _ = self.selfmha_list[i](neighbor_each, neighbor_each, neighbor_each) # [N, B, D]
            neighbor_queryatt, neighbor_attvec = self.mha_list[i](query_embed.unsqueeze(0), neighbor_coatt, neighbor_coatt) # [1, B, D]
            neighbor_queryatt_list.append(neighbor_queryatt.squeeze(0)) # [B, D]
            neighbor_attvec_list.append(neighbor_attvec)
 
        neighbor_queryatt_mat = torch.stack(neighbor_queryatt_list) # [K, B, D]
        neighbor_queryatt_mat = neighbor_queryatt_mat.permute(1, 0, 2) # [B, K, D]
        
        final_repr, final_att = self.finall_addatt(torch.cat((neighbor_queryatt_mat, query_embed.unsqueeze(1)), dim=1)) # [B, K+1, D] -> [B, D]
        logit = self.classifier(final_repr).squeeze() # [B]

        if recon_target != None and mask != None:
            mask_vec = torch.tensor(mask).to(final_repr.device)
            reconstructed = self.recons_layer(final_repr)  
            loss = self.recon_criterion(reconstructed * mask_vec, recon_target * mask_vec)
            return loss
        
        
        if labels != None:
            labels = labels.type_as(logit)
            loss = self.criterion(logit, labels)
            return loss, F.sigmoid(logit)
        
        if attention:
            return F.sigmoid(logit), final_att, neighbor_attvec_list
        
        return F.sigmoid(logit)
    

class AdditiveAttention(torch.nn.Module):
    def __init__(self, in_dim=100, v_size=200):
        super().__init__()

        self.in_dim = in_dim
        self.v_size = v_size
        self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
        self.proj_v = nn.Linear(self.v_size, 1)

    def forward(self, context):
        weights = self.proj_v(self.proj(context)).squeeze(-1)
        weights = torch.softmax(weights, dim=-1) # [B, seq_len]
        return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights # [B, 1, seq_len], [B, seq_len, dim]