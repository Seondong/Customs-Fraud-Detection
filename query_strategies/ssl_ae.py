import numpy as np
import pandas as pd
import pickle
import sys
import gc
import math
import time
import pdb
import argparse
import copy
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

from ranger import Ranger
from utils import find_best_threshold, metrics, torch_threshold


from xgboost import XGBClassifier

import torch
import torchfile
from torch import nn
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

from .strategy import Strategy


class Mish(nn.Module):
    """ Mish: A Self Regularized Non-Monotonic Activation Function 
    https://arxiv.org/abs/1908.08681 """
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))

    
class Dense(nn.Module):
    def __init__(self,input_dim,output_dim,act = Mish()):
        super(Dense,self).__init__()
        self.lin = nn.Linear(input_dim,output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.act = act

    def forward(self,x,use_act=True):
        if use_act:
            output = self.act(self.bn(self.lin(x)))
        else:
            output = self.bn(self.lin(x))

        return output
    

class Encoder(nn.Module):
    def __init__(self, input_dim, layers):
        super(Encoder, self).__init__()
        self.input_layer = Dense(input_dim, layers[0])
        self.hidden_layer = nn.ModuleList([Dense(i,j) for i,j in zip(layers[:-1],layers[1:])])
    
    def forward(self, x):
        dense_vec = self.input_layer(x)
        for idx, layer in enumerate(self.hidden_layer):
            if idx == len(self.hidden_layer)-1:
                dense_vec = layer(dense_vec, False)
            else:
                dense_vec = layer(dense_vec)
        return dense_vec
    
    
class Decoder(nn.Module):
    def __init__(self, input_dim, layers):
        super(Decoder, self).__init__()
        layers = layers[::-1]
        self.output_layer = nn.Linear(layers[-1], input_dim)
        self.hidden_layer = nn.ModuleList([Dense(i,j) for i,j in zip(layers[:-1],layers[1:])])
        
    def forward(self, x):
        for idx, layer in enumerate(self.hidden_layer):
            x = layer(x)
        reconstruct_vec = torch.sigmoid(self.output_layer(x))
        return reconstruct_vec    
    
    
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, layers):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, layers)
        self.decoder = Decoder(input_dim, layers)
        
    def forward(self, x):
        code = self.encode(x)
        output = self.decode(code)
        return output
    
    def encode(self, inp):
        return self.encoder(inp)
    
    def decode(self, code):
        return self.decoder(code)
    
    
class SemiAutoEncoder(nn.Module):
    """ Semi-autoencoder skeleton architecture"""
    def __init__(self, input_dim, layers, device, sigma=0.1):
        super(SemiAutoEncoder, self).__init__()
        self.ae = AutoEncoder(input_dim, layers)
        self.cls_layer = nn.Linear(layers[-1], 1)
        self.sigma = sigma
        self.device = device
        
    def forward(self, x):
        code = self.ae.encode(x)
        output = self.ae.decode(code)
        sup_output = torch.sigmoid(self.cls_layer(code))
        return output, sup_output


def warm_up(epoch, max_epoch=10, w_max=0.1):
    temperature = epoch / max_epoch
    w = np.exp(-5*(1-temperature)**2)*w_max
    return w
    

      
class SSLAutoencoderSampling(Strategy):
    """ Training and fraud detection by the proposed semi-autoencoder model"""
    def __init__(self, args):
        self.args = args
        self.identifier = args.identifier
        super(SSLAutoencoderSampling,self).__init__(data, args)
    
    
    def train_xgb_model(self):
        """ Train XGB model, here we use SMOTE oversampling before fitting a XGB model. """
        print("Training XGBoost model...")
        tmpcol = self.data.dftrainx_lab.columns
        sm = SMOTE(random_state=42)
        self.data.oversampled_trainx, self.data.oversampled_trainy = sm.fit_resample(self.data.dftrainx_lab,self.data.train_cls_label)
        self.data.oversampled_trainx = pd.DataFrame(data=self.data.oversampled_trainx,columns=tmpcol)
        self.xgb = XGBClassifier(n_estimators=100, max_depth=4, n_jobs=-1)
        self.xgb.fit(self.data.oversampled_trainx, self.data.oversampled_trainy)
        
        if self.args.save:
            self.xgb.get_booster().dump_model('./intermediary/xgb_models/xgb_model-readable-'+self.args.identifier+'.txt', with_stats=False)
            self.xgb.get_booster().save_model('./intermediary/xgb_models/xgb_model-'+self.args.identifier+'.json')
        
    
    def prepare_SSL_input(self):
        """ Prepare input for Dual-Attentive Tree-Aware Embedding model, DATE """
         
        # Get leaf index from xgboost model 
        X_train_leaves = self.xgb.apply(self.data.dftrainx_lab)
        X_train_unlab_leaves = self.xgb.apply(self.data.dftrainx_unlab)
        X_valid_leaves = self.xgb.apply(self.data.dfvalidx_lab)
#         X_valid_unlab_leaves = self.xgb.apply(self.data.dfvalidx_unlab)
        X_test_leaves = self.xgb.apply(self.data.dftestx)
        X_leaves = np.concatenate((X_train_leaves, X_train_unlab_leaves, X_valid_leaves), axis=0)
        
        xgbenc = OneHotEncoder(categories="auto")
        xgbenc.fit(X_leaves)
        label_onehot = xgbenc.transform(X_train_leaves)
        unlabel_onehot = xgbenc.transform(X_train_unlab_leaves)
        valid_onehot = xgbenc.transform(X_valid_leaves)
        test_onehot = xgbenc.transform(X_test_leaves)
        
        # calculate repeat times for labeled data
        times = int(np.ceil(unlabel_onehot.shape[0] * 1. / label_onehot.shape[0]))
        
         # create tensordataset
        label_dataset = Data.TensorDataset(torch.FloatTensor(label_onehot.todense()).repeat(times,1), torch.FloatTensor(self.data.train_cls_label).repeat(times))
        unlabel_dataset = Data.TensorDataset(torch.FloatTensor(unlabel_onehot.todense()))
        valid_dataset = Data.TensorDataset(torch.FloatTensor(valid_onehot.todense()), torch.FloatTensor(self.data.valid_cls_label), torch.FloatTensor(self.data.norm_revenue_valid))
        test_dataset = Data.TensorDataset(torch.FloatTensor(test_onehot.todense()), torch.FloatTensor(self.data.test_cls_label), torch.FloatTensor(self.data.norm_revenue_test))
        
        batch_size = 256
        self.data.label_loader = Data.DataLoader(
            dataset=label_dataset,     
            batch_size=batch_size,      
            shuffle=False,               
        )
        
        self.data.iter_label_loader = self.data.label_loader.__iter__()
        
        self.data.unlabel_loader = Data.DataLoader(
            dataset=unlabel_dataset,     
            batch_size=batch_size,      
            shuffle=True,               
        )                                             
        self.data.valid_loader = Data.DataLoader(
            dataset=valid_dataset,     
            batch_size=batch_size,      
            shuffle=False,               
        )
        self.data.test_loader = Data.DataLoader(
            dataset=test_dataset,     
            batch_size=batch_size,      
            shuffle=False,               
        )
    
    def train_model(self):
        """ Get trained model """
        
        input_dim = self.data.unlabel_loader.dataset.tensors[0].shape[-1]
        layers = [512, 256, 20]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SemiAutoEncoder(input_dim, layers, device=device).to(device)
        self.model = nn.DataParallel(self.model)
        
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        optimizer = Ranger(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.cls_objective = nn.BCELoss(reduction="sum")
        self.loss_function = nn.BCELoss(reduction="sum")
        self.best_model = None
        self.global_best_score = 0
        w_max = 0.1 # Max weight for unsupervise loss
        
    
        def train(epoch,max_epoch=10):
            self.model.train()
            train_recon_loss = 0
            train_cls_loss = 0
            for batch_idx, unlab_data in enumerate(self.data.unlabel_loader):
                unlab_data = unlab_data[0].to(device)
                try:
                    label_data, label = next(self.data.iter_label_loader)
                except:
                    self.data.iter_label_loader = self.data.label_loader.__iter__()
                    label_data, label = next(self.data.iter_label_loader)
                label_data, label = label_data.to(device), label.to(device)
                optimizer.zero_grad()
                recon_batch_unlab, _= self.model(unlab_data)
                recon_batch_lab, cls_batch= self.model(label_data)
                recon_batch, data = torch.cat((recon_batch_unlab,recon_batch_lab),dim=0),\
                                                torch.cat((unlab_data,label_data),dim=0)
                recon_loss = warm_up(epoch,max_epoch,w_max=w_max) * self.loss_function(recon_batch, data) / 100
                cls_loss = self.cls_objective(cls_batch.view(-1,1),label.view(-1,1))
                loss = recon_loss + cls_loss
                loss.backward()
                train_recon_loss += recon_loss.item()
                train_cls_loss += cls_loss.item()
                optimizer.step()
                if batch_idx % 500 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tRECON Loss: {:.6f} CLS Loss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.data.unlabel_loader.dataset),
                        100. * batch_idx / len(self.data.unlabel_loader),
                        recon_loss.item() / len(data), cls_loss.item() / len(data)))

            print('====> Epoch: {} Average RECON Loss: {:.4f} CLS Loss: {:.4f}'.format(
                  epoch, train_recon_loss / len(self.data.unlabel_loader.dataset), train_cls_loss / len(self.data.unlabel_loader.dataset)))
        
        
        def valid(epoch):
            global state_dict
            
            self.model.eval()
            recon_loss = 0
            cls_loss = 0
            pred_prob = []
            with torch.no_grad():
                for i, (data,label,rev) in enumerate(self.data.valid_loader):
                    data = data.to(device)
                    label = label.to(device)
                    recon_batch, cls_batch = self.model(data)
                    recon_loss += self.loss_function(recon_batch, data).item()
                    cls_loss += self.cls_objective(cls_batch.view(-1,1),label.view(-1,1)).item()
                    cls_batch = cls_batch.detach().cpu().numpy().tolist()
                    pred_prob.extend(cls_batch)

            recon_loss /= len(self.data.valid_loader.dataset)
            cls_loss /= len(self.data.valid_loader.dataset)
            print('====> Valid set RECON Loss: {:.4f} CLS Loss: {:.4f}'.format(recon_loss,cls_loss))

            y_prob = np.ravel(pred_prob)
            xgb_validy = self.data.valid_loader.dataset.tensors[1]
            revenue_valid = self.data.valid_loader.dataset.tensors[2]
            
            overall_f1, auc, precisions, re, f1s, rev = metrics(y_prob,xgb_validy,revenue_valid,self.args)
            select_best = np.mean(rev) 

            print("Over-all F1:%.4f, AUC:%.4f, F1-top:%.4f" % (overall_f1, auc, select_best) )

            # save best model 
            if select_best >= self.global_best_score:
                self.global_best_score = select_best
                state_dict = self.model.state_dict()
                self.best_model = copy.deepcopy(self.model)

        
        self.max_epoch = self.args.epoch
        for epoch in range(self.max_epoch): 
            train(epoch, self.max_epoch)
            valid(epoch)
    
    
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        self.best_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        recon_loss = 0
        cls_loss = 0

        pred_prob = []
        with torch.no_grad():
            for i, (data,label,rev) in enumerate(self.data.test_loader):
                data = data.to(device)
                label = label.to(device)
                recon_batch, cls_batch = self.best_model(data)
                recon_loss += self.loss_function(recon_batch, data).item()
                cls_loss += self.cls_objective(cls_batch.view(-1,1),label.view(-1,1)).item()
                cls_batch = cls_batch.detach().cpu().numpy().tolist()
                pred_prob.extend(cls_batch)

        recon_loss /= len(self.data.test_loader.dataset)
        cls_loss /= len(self.data.test_loader.dataset)
        print('====> Test set RECON Loss: {:.4f} CLS Loss: {:.4f}'.format(recon_loss,cls_loss))

        self.y_prob = np.ravel(pred_prob)

    
    def query(self, k):
        self.train_xgb_model()
        self.prepare_SSL_input()
        self.train_model()
        self.predict_frauds()
        chosen = np.argpartition(self.y_prob[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()

