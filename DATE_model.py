import argparse
import os
import pickle
import warnings
import time 
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from model.AttTreeEmbedding import Attention, DATE
from ranger import Ranger
from utils import torch_threshold, metrics

warnings.filterwarnings("ignore")



        
class VanillaDATE:
    
    def __init__(self, data, curr_time, state_dict = None):
        self.data = data
        self.curr_time = curr_time
        self.state_dict = state_dict
        self.model_name = "DATE"
        self.model_path = "./intermediary/saved_models/%s-%s.pkl" % (self.model_name,self.curr_time)
        
    def train(self, args):
        # get data
        train_loader, valid_loader, test_loader, leaf_num, importer_size, item_size, xgb_validy, xgb_testy, revenue_valid, revenue_test = self.data
        # get configs
        epochs = args.epoch
        dim = args.dim
        lr = args.lr
        weight_decay = args.l2
        head_num = args.head_num
        act = args.act
        fusion = args.fusion
        beta = args.beta
        alpha = args.alpha
        use_self = args.use_self
        agg = args.agg
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DATE(leaf_num,importer_size,item_size,\
                                        dim,head_num,\
                                        fusion_type=fusion,act=act,device=device,\
                                        use_self=use_self,agg_type=agg,
                                        ).to(device)
        
        if torch.cuda.device_count() > 1:
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

        cls_loss_func = nn.BCELoss()
        reg_loss_func = nn.MSELoss()

        # save best model
        global_best_score = 0
        model_state = None

        # early stop settings 
        stop_rounds = 3
        no_improvement = 0
        current_score = None 
        
        print("Running the DATE model ...")
        for epoch in range(epochs):
            for step, (batch_feature,batch_user,batch_item,batch_cls,batch_reg) in enumerate(train_loader):
                self.model.train() # prep to train model
                batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
                batch_feature.to(device), batch_user.to(device), batch_item.to(device),\
                 batch_cls.to(device), batch_reg.to(device)
                batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)

                # model output
                classification_output, regression_output, hidden_vector = self.model(batch_feature,batch_user,batch_item)

                cls_loss = cls_loss_func(classification_output,batch_cls)
                revenue_loss = alpha * reg_loss_func(regression_output, batch_reg)
                loss = cls_loss + revenue_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (step+1) % 1000 ==0:  
                    print("CLS loss:%.4f, REG loss:%.4f, ADV loss:%.4f, Loss:%.4f"\
                    %(cls_loss.item(),revenue_loss.item(),adv_loss.item(),loss.item()))
                    
            # evaluate 
            self.model.eval()
            print("------------")
            print("Validate at epoch %s"%(epoch+1))
            y_prob, val_loss, _ = self.model.module.eval_on_batch(valid_loader)
            y_pred_tensor = torch.tensor(y_prob).float().to(device)
            best_threshold, val_score, roc = torch_threshold(y_prob,xgb_validy)
            overall_f1, auc, precisions, recalls, f1s, revenues = metrics(y_prob,xgb_validy,revenue_valid)
            select_best = np.mean(revenues)  # instead of f1s
            print("Over-all F1:%.4f, AUC:%.4f, F1-top:%.4f" % (overall_f1, auc, select_best) )

            print("Evaluate at epoch %s"%(epoch+1))
            y_prob, val_loss, _ = self.model.module.eval_on_batch(test_loader)
            y_pred_tensor = torch.tensor(y_prob).float().to(device)
            overall_f1, auc, precisions, recalls, f1s, revenues = metrics(y_prob,xgb_testy,revenue_test, best_thresh=best_threshold)
            print("Over-all F1:%.4f, AUC:%.4f, F1-top:%.4f" %(overall_f1, auc, np.mean(f1s)) )

            # save best model 
            if select_best > global_best_score:
                global_best_score = select_best
                torch.save(self.model, self.model_path)
            
            # early stopping 
            if current_score == None:
                current_score = select_best
                continue
            if select_best < current_score:
                current_score = select_best
                no_improvement += 1
            if no_improvement >= stop_rounds:
                print("Early stopping...")
                break 
            if select_best > current_score:
                no_improvement = 0
                current_score = None

    def evaluate(self):
        #get data
        train_loader, valid_loader, test_loader, leaf_num, importer_size, item_size, xgb_validy, xgb_testy, revenue_valid, revenue_test = self.data
        print()
        print("--------Evaluating DATE model---------")
        # create best model
        best_model = torch.load(self.model_path)
        best_model.eval()

        # get threshold
        y_prob, val_loss, _ = best_model.module.eval_on_batch(valid_loader)
        best_threshold, val_score, roc = torch_threshold(y_prob,xgb_validy)

        # predict test 
        y_prob, val_loss, _ = best_model.module.eval_on_batch(test_loader)
        overall_f1, auc, precisions, recalls, f1s, revenues = metrics(y_prob,xgb_testy,revenue_test,best_threshold)
        best_score = f1s[0]
        
        return overall_f1, auc, precisions, recalls, f1s, revenues, self.model_path