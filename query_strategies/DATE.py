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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from ranger import Ranger
from .strategy import Strategy
from xgboost import XGBClassifier
from torch.utils.data import DataLoader
from utils import find_best_threshold, process_leaf_idx, torch_threshold, metrics
from model.AttTreeEmbedding import Attention, DATEModel
from model.utils import FocalLoss



class DATESampling(Strategy):
    """ DATE strategy: Using DATE classification probability to measure fraudness of imports 
        Reference: DATE: Dual Attentive Tree-aware Embedding for Customs Fraud Detection; KDD 2020 """
    
    def __init__(self, args):
        super(DATESampling,self).__init__(args)
        self.model_name = "DATE"
        self.model_path = "./intermediary/saved_models/%s-%s.pkl" % (self.model_name,self.args.identifier)
        self.batch_size = args.batch_size
        
    
    def train_xgb_model(self):
        """ Train XGB model """
        print("Training XGBoost model...")
        self.xgb = XGBClassifier(n_estimators=100, max_depth=4, n_jobs=-1)
        self.xgb.fit(self.data.dftrainx_lab, self.data.train_cls_label)     
        
        if self.args.save:
            self.xgb.get_booster().dump_model('./intermediary/xgb_models/xgb_model-readable-'+self.args.identifier+'.txt', with_stats=False)
            self.xgb.get_booster().save_model('./intermediary/xgb_models/xgb_model-'+self.args.identifier+'.json')
        
    
    def prepare_DATE_input(self):
        """ Prepare input for Dual-Attentive Tree-Aware Embedding model, DATE """
                
        # user & item information 
        if self.data.args.data != 'synthetic-k':
            importer_id = 'importer.id'
            tariff_code = 'tariff.code'
        else:
            importer_id = '수입자부호'
            tariff_code = 'HS10단위부호'

        train_raw_importers = self.data.train_lab[importer_id].values
        train_raw_items = self.data.train_lab[tariff_code].values
        valid_raw_importers = self.data.valid_lab[importer_id].values
        valid_raw_items = self.data.valid_lab[tariff_code].values
        test_raw_importers = self.data.test[importer_id]
        test_raw_items = self.data.test[tariff_code]

        # we need padding for unseen user or item 
        importer_set = set(train_raw_importers)
        item_set = set(train_raw_items)

        # Remember to +1 for zero padding 
        importer_mapping = {v:i+1 for i,v in enumerate(importer_set)} 
        hs6_mapping = {v:i+1 for i,v in enumerate(item_set)}
        self.data.importer_size = len(importer_mapping) + 1
        self.data.item_size = len(hs6_mapping) + 1
        train_importers = [importer_mapping[x] for x in train_raw_importers]
        train_items = [hs6_mapping[x] for x in train_raw_items]

        # for test data, we use padding_idx=0 for unseen data
        valid_importers = [importer_mapping.get(x,0) for x in valid_raw_importers]
        valid_items = [hs6_mapping.get(x,0) for x in valid_raw_items]
        test_importers = [importer_mapping.get(x,0) for x in test_raw_importers] # use dic.get(key,deafault) to handle unseen
        test_items = [hs6_mapping.get(x,0) for x in test_raw_items]

        # Get leaf index from xgboost model 
        X_train_leaves = self.xgb.apply(self.data.dftrainx_lab)
        X_valid_leaves = self.xgb.apply(self.data.dfvalidx_lab)
        X_test_leaves = self.xgb.apply(self.data.dftestx)
        
        # Preprocess
        train_rows = self.data.train_lab.shape[0]
        valid_rows = self.data.valid_lab.shape[0] + train_rows
        X_leaves = np.concatenate((X_train_leaves, X_valid_leaves, X_test_leaves), axis=0) # make sure the dimensionality
        transformed_leaves, self.data.leaf_num, new_leaf_index = process_leaf_idx(X_leaves)
        train_leaves, valid_leaves, test_leaves = transformed_leaves[:train_rows],\
                                                  transformed_leaves[train_rows:valid_rows],\
                                                  transformed_leaves[valid_rows:]

        # Convert to torch type
        train_leaves = torch.tensor(train_leaves).long()
        train_user = torch.tensor(train_importers).long()
        train_item = torch.tensor(train_items).long()

        valid_leaves = torch.tensor(valid_leaves).long()
        valid_user = torch.tensor(valid_importers).long()
        valid_item = torch.tensor(valid_items).long()

        test_leaves = torch.tensor(test_leaves).long()
        test_user = torch.tensor(test_importers).long()
        test_item = torch.tensor(test_items).long()

        # cls data
        train_label_cls = torch.tensor(self.data.train_cls_label).float()
        valid_label_cls = torch.tensor(self.data.valid_cls_label).float()
        test_label_cls = torch.tensor(self.data.test_cls_label).float()

        # revenue data 
        train_label_reg = torch.tensor(self.data.norm_revenue_train).float()
        valid_label_reg = torch.tensor(self.data.norm_revenue_valid).float()
        test_label_reg = torch.tensor(self.data.norm_revenue_test).float()

        train_dataset = Data.TensorDataset(train_leaves,train_user,train_item,train_label_cls,train_label_reg)
        valid_dataset = Data.TensorDataset(valid_leaves,valid_user,valid_item,valid_label_cls,valid_label_reg)
        test_dataset = Data.TensorDataset(test_leaves,test_user,test_item,test_label_cls,test_label_reg)
        
        
        self.data.train_loader = Data.DataLoader(
            dataset=train_dataset,     
            batch_size=self.batch_size,      
            shuffle=True
        )
        self.data.valid_loader = Data.DataLoader(
            dataset=valid_dataset,     
            batch_size=self.batch_size,      
            shuffle=False
        )
        self.data.test_loader = Data.DataLoader(
            dataset=test_dataset,     
            batch_size=self.batch_size,      
            shuffle=False
        )
        
        
           # save data
        if self.args.save:
            data4embedding = {"train_dataset":train_dataset,"valid_dataset":valid_dataset,"test_dataset":test_dataset,\
                      "leaf_num":self.data.leaf_num,"importer_num":self.data.importer_size,"item_size":self.data.item_size}
            with open("./intermediary/torch_data/torch_data-"+self.args.identifier+".pickle", 'wb') as f:
                pickle.dump(data4embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open("./intermediary/leaf_indices/leaf_index-"+self.args.identifier+".pickle", "wb") as f:
                pickle.dump(new_leaf_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    def get_model(self):
        return torch.load(self.model_path)
    
    
    def train_DATE_model(self):
        """ Train DATE model """
        
        print(f'Mode: {self.args.mode}, Episode: {self.data.episode}')
        
        if self.args.mode == 'scratch' or self.data.episode == 0:
            self.date_model = VanillaDATE(self.data, self.args)
        else:
            self.date_model = self.get_model()
            self.date_model = VanillaDATE(self.data, self.args, self.date_model.state_dict())
        
        self.date_model.train(self.args)
        overall_f1, auc, precisions, recalls, f1s, revenues, self.model_path = self.date_model.evaluate()
            
    
    def predict_frauds(self):
        """ Prediction for new dataset (test_model) """
        best_model = self.get_model()
        final_output, _, (hiddens, revs) = best_model.module.eval_on_batch(self.data.test_loader)
        self.y_prob = final_output
        
        
    def query(self, k, model_available = False):
        if not model_available:
            self.train_xgb_model()
            self.prepare_DATE_input()
            self.train_DATE_model()
        self.predict_frauds()
        chosen = np.argpartition(self.y_prob[self.available_indices], -k)[-k:]
        return self.available_indices[chosen].tolist()
    
    
    
    # Below methods are for DATE-dependent selection strategies.

    def get_embedding_valid(self):
        best_model = self.get_model()
        final_output, _, (hiddens, revs) = best_model.module.eval_on_batch(self.data.valid_loader)
        # hiddens = [hiddens[i] for i in self.available_indices_valid]
        return hiddens


    def get_embedding_test(self):
        best_model = self.get_model()
        final_output, _, (hiddens, revs) = best_model.module.eval_on_batch(self.data.test_loader)
        # hiddens = [hiddens[i] for i in self.available_indices]
        return hiddens
    
    
    def rev_score(self):
        if self.rev_func == 'log':
            return lambda x: math.log(2+x)
        return lambda x: x
    
    
    def get_output(self):
        best_model = self.get_model()
        final_output, _, (hiddens, revs) = best_model.module.eval_on_batch(self.data.test_loader)
        return final_output[self.available_indices]


    def get_revenue(self):
        best_model = self.get_model()
        final_output, _, (hiddens, revs) = best_model.module.eval_on_batch(self.data.test_loader)
        revs = [revs[i] for i in self.available_indices]
        return revs

    
    def get_grad_embedding(self):
        embDim = self.args.dim
        best_model = torch.load(self.model_path)
        final_output, _, (hiddens, revs) = best_model.module.eval_on_batch(self.data.test_loader)
        nLab = 2
        print(len(final_output), hiddens[0].shape, len(hiddens))
        embedding = np.zeros([self.num_data, embDim * nLab])
        with torch.no_grad():
            for idx, prob in enumerate(final_output):
                maxInds = np.asarray([0, 0])
                probs = np.asarray([1 - prob, prob])
                if prob >= 0.5:
                    maxInd = 1
                else:
                    maxInd = 0
                if self.args.device == 'cpu':
                    for c in range(nLab):
                        if c == maxInd:
                            embedding[idx][embDim * c : embDim * (c+1)] = (hiddens[idx] * (1 - probs[c]))
                        else:
                            embedding[idx][embDim * c : embDim * (c+1)] = (hiddens[idx] * (0 - probs[c]))
                else:
                    for c in range(nLab):
                        if c == maxInd:
                            embedding[idx][embDim * c : embDim * (c+1)] = (hiddens[idx] * (1 - probs[c])).cpu().numpy()
                        else:
                            embedding[idx][embDim * c : embDim * (c+1)] = (hiddens[idx] * (0 - probs[c])).cpu().numpy()
            return embedding[self.available_indices]


    
    
class VanillaDATE:
    
    def __init__(self, data, args, state_dict = None):
        self.data = data
        self.args = args
        self.state_dict = state_dict
        self.model_name = "DATE"
        self.model_path = "./intermediary/saved_models/%s-%s.pkl" % (self.model_name,self.args.identifier)
        
        
    def train(self, args):
        
        train_loader = self.data.train_loader
        valid_loader = self.data.valid_loader
        test_loader = self.data.test_loader
        leaf_num = self.data.leaf_num 
        importer_size = self.data.importer_size 
        item_size = self.data.item_size 
        xgb_validy = self.data.valid_cls_label
        xgb_testy = self.data.test_cls_label 
        revenue_valid = self.data.valid_reg_label 
        revenue_test = self.data.test_reg_label 
        
        # get configs
        epochs = args.epoch
        dim = args.dim
        lr = args.lr
        weight_decay = args.l2
        head_num = args.head_num
        act = args.act
        fusion = args.fusion
        alpha = args.alpha
        use_self = args.use_self
        agg = args.agg
        closs = args.closs
        rloss = args.rloss
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DATEModel(leaf_num,importer_size,item_size,\
                                        dim,head_num,\
                                        fusion_type=fusion,act=act,device=device,\
                                        use_self=use_self,agg_type=agg, cls_loss_func=closs, reg_loss_func=rloss).to(device)
        
#         if torch.cuda.device_count() > 1:
        self.model = nn.DataParallel(self.model)     
            
        if not self.state_dict:
            # initialize parameters
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            # filter out unnecessary keys
            self.state_dict.pop('module.leaf_embedding.weight')
            self.state_dict.pop('module.user_embedding.weight')
            self.state_dict.pop('module.item_embedding.weight')
            state = self.model.state_dict()
            state.update(self.state_dict)
            self.model.load_state_dict(state)
        # optimizer & loss 
        optimizer = Ranger(self.model.parameters(), weight_decay=weight_decay,lr=lr)
        
        self.cls_loss_func = closs
        self.reg_loss_func = rloss

        # save best model
        global_best_score = 0
        self.performance = 0
        model_state = None

        # early stop settings 
        stop_rounds = 3
        no_improvement = 0
        current_score = None 
        
        print()
        print("Training DATE model ...")
        for epoch in range(epochs):
            for step, (batch_feature,batch_user,batch_item,batch_cls,batch_reg) in enumerate(train_loader):
                self.model.train() # prep to train model
                batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
                batch_feature.to(device), batch_user.to(device), batch_item.to(device),\
                 batch_cls.to(device), batch_reg.to(device)
                batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)

                # model output
                classification_output, regression_output, hidden_vector = self.model(batch_feature,batch_user,batch_item)
                
                if self.cls_loss_func == 'bce':
                    cls_loss = nn.BCELoss()(classification_output,batch_cls)
                if self.cls_loss_func == 'focal':
                    cls_loss = FocalLoss()(classification_output, batch_cls)

                # compute regression loss 
                if self.reg_loss_func == 'full':
                    revenue_loss = nn.MSELoss()(regression_output, batch_reg)
                if self.reg_loss_func == 'masked':
                    revenue_loss = torch.mean(nn.MSELoss(reduction = 'none')(regression_output, batch_reg)*batch_cls)

                loss = cls_loss + alpha*revenue_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (step+1) % 1000 ==0:  
                    print("CLS loss:%.4f, REG loss:%.4f, Loss:%.4f"\
                    %(cls_loss.item(),revenue_loss.item(),loss.item()))
                    
            # evaluate 
            self.model.eval()
            print("------------")
            print("Validate at epoch %s"%(epoch+1))
            y_prob, val_loss, _ = self.model.module.eval_on_batch(valid_loader)
            y_pred_tensor = torch.tensor(y_prob).float().to(device)
            best_threshold, val_score, roc = torch_threshold(y_prob,xgb_validy)
            overall_f1, auc, precisions, recalls, f1s, revenues = metrics(y_prob,xgb_validy,revenue_valid,self.args)
            select_best = np.mean(precisions+revenues)  # instead of f1s
            print("Overall F1:%.4f, AUC:%.4f, F1-top:%.4f" % (overall_f1, auc, select_best))

            # save best model 
            if select_best >= global_best_score:
                global_best_score = select_best
                self.model.module.performance = np.mean(revenues)
                torch.save(self.model, self.model_path)
                print(os.path.abspath(self.model_path))
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= stop_rounds:
                print("Early stopping...")
                break 
                
                
    def evaluate(self):
        
        train_loader = self.data.train_loader
        valid_loader = self.data.valid_loader
        test_loader = self.data.test_loader
        leaf_num = self.data.leaf_num 
        importer_size = self.data.importer_size 
        item_size = self.data.item_size 
        xgb_validy = self.data.valid_cls_label
        xgb_testy = self.data.test_cls_label 
        revenue_valid = self.data.valid_reg_label 
        revenue_test = self.data.test_reg_label 
        
        print()
        print("--------Evaluating DATE model---------")
        # create best model
        best_model = torch.load(self.model_path)
        best_model.eval()

        # get threshold
        y_prob, val_loss, _ = best_model.module.eval_on_batch(valid_loader)
        best_threshold, val_score, roc = torch_threshold(y_prob,xgb_validy)

        # predict test 
        y_prob, val_loss, (hidden, revs) = best_model.module.eval_on_batch(test_loader)
        overall_f1, auc, precisions, recalls, f1s, revenues = metrics(y_prob,xgb_testy,revenue_test,best_threshold)
        best_score = f1s[0]
        
        return overall_f1, auc, precisions, recalls, f1s, revenues, self.model_path    
    