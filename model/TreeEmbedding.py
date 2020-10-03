import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
import numpy as np 
from .utils import FocalLoss

class MultiTreeEmbeddingClassifier(nn.Module):
    def __init__(self,max_leaf,dim,forest_pooling="max",device="cuda:0", cls_loss_func = 'bce', reg_loss_func = 'full'):
        super(MultiTreeEmbeddingClassifier, self).__init__()
        self.d = dim
        self.device = device
        self.leaf_embedding = nn.Embedding(max_leaf,dim)
        self.hidden = nn.Linear(dim,dim)
        self.output_cls_layer = nn.Linear(dim,1)
        self.output_reg_layer = nn.Linear(dim,1)
        self.pooling = forest_pooling
        self.cls_loss_func = cls_loss_func
        self.reg_loss_func = reg_loss_func

    def forward(self,x):
        leaf_vectors = self.leaf_embedding(x)

        if self.pooling =="max": # max pooling
            set_vector,_ = torch.max(leaf_vectors,dim=1)
        elif self.pooling =="mean": # mean pooling 
            set_vector = torch.mean(leaf_vectors,dim=1)
        else:
            raise NotImplementedError

        hidden = self.hidden(set_vector)
        hidden = F.leaky_relu(hidden)
        classification_output = torch.sigmoid(self.output_cls_layer(hidden))
        regression_output = self.output_reg_layer(hidden)
        return classification_output, regression_output
    
    def eval_on_batch(self,test_loader): # predict test data using batch 
        final_output = []
        cls_loss = []
        reg_loss = []

        for batch in test_loader:
            batch_feature, batch_user, batch_item, batch_cls, batch_reg = batch
            batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
            batch_feature.to(self.device), batch_user.to(self.device),\
            batch_item.to(self.device), batch_cls.to(self.device), batch_reg.to(self.device)
            batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)
            y_pred_prob, y_pred_rev = self.forward(batch_feature)

            # compute classification loss
            if self.cls_loss_func == 'bce':
                cls_losses = nn.BCELoss()(y_pred_prob,batch_cls)
            if self.cls_loss_func == 'focal':
                cls_losses = FocalLoss()(y_pred_prob,batch_cls)

            cls_loss.append(cls_losses.item())
            # compute regression loss 
            if self.reg_loss_func == 'full':
                reg_losses = nn.MSELoss()(y_pred_rev, batch_reg)
            if self.reg_loss_func == 'mask':
                reg_losses = torch.mean(nn.MSELoss(reduction = 'none')(y_pred_rev, batch_reg)*batch_cls)
            
            cls_loss.append(cls_losses.item())
            reg_loss.append(reg_losses.item())

            # store predicted probability 
            y_pred = y_pred_prob.detach().cpu().numpy().tolist()
            final_output.extend(y_pred)

        print("Testing...\n CLS loss: %.4f, REG loss: %.4f"% (np.mean(cls_loss), np.mean(reg_loss)) )
        return np.array(final_output).ravel(), np.mean(cls_loss)+ np.mean(reg_loss)