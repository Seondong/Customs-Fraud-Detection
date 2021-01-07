import torch
from pytorch_memlab import LineProfiler
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
import numpy as np 
from torch_multi_head_attention import MultiHeadAttention
from .utils import FocalLoss
import gc
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score


class Mish(nn.Module):
    """ Mish: A Self Regularized Non-Monotonic Activation Function 
    https://arxiv.org/abs/1908.08681 """
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))


class FusionAttention(nn.Module):
    def __init__(self,dim):
        super(FusionAttention, self).__init__()
        self.attention_matrix = nn.Linear(dim, dim)
        self.project_weight = nn.Linear(dim,1)
    def forward(self, inputs):
        query_project = self.attention_matrix(inputs) # (b,t,d) -> (b,t,d2)
        query_project = F.leaky_relu(query_project)
        project_value = self.project_weight(query_project) # (b,t,h) -> (b,t,1)
        attention_weight = torch.softmax(project_value, dim=1) # Normalize and calculate weights (b,t,1)
        attention_vec = inputs * attention_weight
        attention_vec = torch.sum(attention_vec,dim=1)
        return attention_vec, attention_weight


class Attention(nn.Module):
    def __init__(self,dim,hidden,aggregate="sum"):
        super(Attention, self).__init__()
        self.attention_matrix = nn.Linear(dim, hidden)
        self.project_weight = nn.Linear(hidden*2,hidden)
        self.h = nn.Parameter(torch.rand(hidden,1))
        self.agg_type = aggregate
    def forward(self, query,key): # assume key==value
        dim = query.size(-1)
        batch,time_step = key.size(0) ,key.size(1)
        
        # concate input query and key 
        query = query.view(batch,1,dim)
        query = query.expand(batch,time_step,-1)
        cat_vector = torch.cat((query,key),dim=-1)
        
        # project to single value
        project_vector = self.project_weight(cat_vector) 
        project_vector = torch.relu(project_vector)
        attention_alpha = torch.matmul(project_vector,self.h)
        attention_weight = torch.softmax(attention_alpha, dim=1) # Normalize and calculate weights (b,t,1)
        attention_vec = key * attention_weight
        
        # aggregate leaves
        if self.agg_type == "max":
            attention_vec,_ = torch.max(attention_vec,dim=1)
        elif self.agg_type =="mean":
            attention_vec = torch.mean(attention_vec,dim=1)
        elif self.agg_type =="sum":
            attention_vec = torch.sum(attention_vec,dim=1)
        return attention_vec, attention_weight


class DATEModel(nn.Module):
    def __init__(self,max_leaf,importer_size,item_size,dim,head_num=4,fusion_type="concat",act="relu",device="cpu",use_self=True,agg_type="sum", cls_loss_func = 'bce', reg_loss_func = 'full'):
        super(DATEModel, self).__init__()
        self.d = dim
        self.device = device
        if act == "relu":
            self.act = nn.LeakyReLU()
        elif act == "mish":
            self.act = Mish() 
        self.fusion_type = fusion_type
        self.use_self = use_self
        self.performance = 0

        # embedding layers 
        self.leaf_embedding = nn.Embedding(max_leaf,dim)
        self.user_embedding = nn.Embedding(importer_size,dim,padding_idx=0)
        self.user_embedding.weight.data[0] = torch.zeros(dim)
        self.item_embedding = nn.Embedding(item_size,dim,padding_idx=0)
        self.item_embedding.weight.data[0] = torch.zeros(dim)

        # attention layer
        self.attention_bolck = Attention(dim,dim,agg_type).to(device)
        self.self_att = MultiHeadAttention(dim,head_num).to(device)
        self.fusion_att = FusionAttention(dim)

        # Hidden & output layer
        self.layer_norm = nn.LayerNorm((100,dim))
        self.fussionlayer = nn.Linear(dim*3,dim)
        self.hidden = nn.Linear(dim,dim)
        self.output_cls_layer = nn.Linear(dim,1)
        self.output_reg_layer = nn.Linear(dim,1)
        
        # Loss function
        self.reg_loss_func = reg_loss_func 
        self.cls_loss_func = cls_loss_func 

    def forward(self,feature,uid,item_id):
        leaf_vectors = self.leaf_embedding(feature)
        if self.use_self:
            leaf_vectors = self.self_att(leaf_vectors,leaf_vectors,leaf_vectors)
            leaf_vectors = self.layer_norm(leaf_vectors)
        importer_vector = self.user_embedding(uid)
        item_vector = self.item_embedding(item_id)
        query_vector = importer_vector * item_vector
        set_vector, self.attention_w = self.attention_bolck(query_vector,leaf_vectors)
        
        # concat the user, item and tree vectors into a fusion embedding
        if self.fusion_type == "concat":
            fusion = torch.cat((importer_vector, item_vector, set_vector), dim=-1)
            fusion = self.act(self.fussionlayer(fusion))
        elif self.fusion_type == "attention":
            importer_vector, item_vector, set_vector = importer_vector.view(-1,1,self.d), item_vector.view(-1,1,self.d), set_vector.view(-1,1,self.d)
            fusion = torch.cat((importer_vector, item_vector, set_vector), dim=1)
            fusion,_ = self.fusion_att(fusion)
        else:
            raise "Fusion type error"
        hidden = self.hidden(fusion)
        hidden = self.act(hidden)

        # multi-task output 
        classification_output = torch.sigmoid(self.output_cls_layer(hidden))
        regression_output = torch.relu(self.output_reg_layer(hidden))
        return classification_output, regression_output, hidden

    def pred_from_hidden(self,hidden):
        classification_output = torch.sigmoid(self.output_cls_layer(hidden))
        return classification_output 

    def eval_on_batch(self,test_loader): # predict test data using batch 
        final_output = []
        cls_loss = []
        reg_loss = []
        hiddens = []
        revs = []  
        i = 0
        
        with torch.no_grad():
            for batch in test_loader:
                i += 1
                batch_feature, batch_user, batch_item, batch_cls, batch_reg = batch
                batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
                batch_feature.to(self.device), batch_user.to(self.device),\
                batch_item.to(self.device), batch_cls.to(self.device), batch_reg.to(self.device)
                batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)
                y_pred_prob, y_pred_rev, hidden = self.forward(batch_feature,batch_user,batch_item)
                revs.extend(y_pred_rev)
                hiddens.extend(hidden)

                # compute classification loss
                if self.cls_loss_func == 'bce':
                    cls_losses = nn.BCELoss()(y_pred_prob,batch_cls)
                if self.cls_loss_func == 'focal':
                    cls_losses = FocalLoss()(y_pred_prob, batch_cls)
                cls_loss.append(cls_losses.item())

                # compute regression loss 
                if self.reg_loss_func == 'full':
                    reg_losses = nn.MSELoss()(y_pred_rev, batch_reg)
                if self.reg_loss_func == 'masked':
                    reg_losses = torch.mean(nn.MSELoss(reduction = 'none')(y_pred_rev, batch_reg)*batch_cls)
                reg_loss.append(reg_losses.item())

                # store predicted probability 
                y_pred = y_pred_prob.detach().cpu().numpy().tolist()
                final_output.extend(y_pred)

                del hidden
                del cls_losses
                del reg_losses
                del y_pred

        print("CLS loss: %.4f, REG loss: %.4f"% (np.mean(cls_loss), np.mean(reg_loss)) )
        return np.array(final_output).ravel(), np.mean(cls_loss)+ np.mean(reg_loss), (hiddens, revs)

class TransferDATEModel(nn.Module):
    def __init__(self,max_leaf,importer_size,item_size,dim,head_num=4,fusion_type="concat",act="relu",device="cpu",use_self=True,agg_type="sum", cls_loss_func = 'bce', reg_loss_func = 'full'):
        super(TransferDATEModel, self).__init__()
        self.d = dim
        self.device = device
        if act == "relu":
            self.act = nn.LeakyReLU()
        elif act == "mish":
            self.act = Mish() 
        self.fusion_type = fusion_type
        self.use_self = use_self
        self.performance = 0

        # embedding layers 
        self.leaf_embedding = nn.Embedding(max_leaf,dim)
        self.user_embedding = nn.Embedding(importer_size,dim,padding_idx=0)
        self.user_embedding.weight.data[0] = torch.zeros(dim)
        self.item_embedding = nn.Embedding(item_size,dim,padding_idx=0)
        self.item_embedding.weight.data[0] = torch.zeros(dim)

        # attention layer
        self.attention_bolck = Attention(dim,dim,agg_type).to(device)
        self.self_att = MultiHeadAttention(dim,head_num).to(device)
        self.fusion_att = FusionAttention(dim)

        # Hidden & output layer
        self.layer_norm = nn.LayerNorm((100,dim))
        self.fussionlayer = nn.Linear(dim*3,dim)
        self.hidden = nn.Linear(dim,dim)
        self.output_cls_layer = nn.Linear(dim,1)
        self.output_reg_layer = nn.Linear(dim,1)
        
        # Loss function
        self.reg_loss_func = reg_loss_func 
        self.cls_loss_func = cls_loss_func 

    def forward(self,feature,uid,item_id):
        leaf_vectors = self.leaf_embedding(feature)
        if self.use_self:
            leaf_vectors = self.self_att(leaf_vectors,leaf_vectors,leaf_vectors)
            leaf_vectors = self.layer_norm(leaf_vectors)
        importer_vector = self.user_embedding(uid)
        item_vector = self.item_embedding(item_id)
        query_vector = importer_vector * item_vector
        set_vector, self.attention_w = self.attention_bolck(query_vector,leaf_vectors)
        
        # concat the user, item and tree vectors into a fusion embedding
        if self.fusion_type == "concat":
            fusion = torch.cat((importer_vector, item_vector, set_vector), dim=-1)
            fusion = self.act(self.fussionlayer(fusion))
        elif self.fusion_type == "attention":
            importer_vector, item_vector, set_vector = importer_vector.view(-1,1,self.d), item_vector.view(-1,1,self.d), set_vector.view(-1,1,self.d)
            fusion = torch.cat((importer_vector, item_vector, set_vector), dim=1)
            fusion,_ = self.fusion_att(fusion)
        else:
            raise "Fusion type error"
        hidden = self.hidden(fusion)
        hidden = self.act(hidden)
        hidden = F.normalize(hidden, dim=1)

        # multi-task output 
        classification_output = torch.sigmoid(self.output_cls_layer(hidden))
        regression_output = torch.relu(self.output_reg_layer(hidden))
        return classification_output, regression_output, hidden

    def pred_from_hidden(self,hidden):
        classification_output = torch.sigmoid(self.output_cls_layer(hidden))
        return classification_output 

    def eval_on_batch(self,test_loader): # predict test data using batch 
        final_output = []
        cls_loss = []
        reg_loss = []
        hiddens = []
        revs = []  
        i = 0
        
        with torch.no_grad():
            for batch in test_loader:
                i += 1
                batch_feature, batch_user, batch_item, batch_cls, batch_reg, _ = batch
                batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
                batch_feature.to(self.device), batch_user.to(self.device),\
                batch_item.to(self.device), batch_cls.to(self.device), batch_reg.to(self.device)
                batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)
                y_pred_prob, y_pred_rev, hidden = self.forward(batch_feature,batch_user,batch_item)
                revs.extend(y_pred_rev)
                hiddens.extend(hidden)

                # compute classification loss
                if self.cls_loss_func == 'bce':
                    cls_losses = nn.BCELoss()(y_pred_prob,batch_cls)
                if self.cls_loss_func == 'focal':
                    cls_losses = FocalLoss()(y_pred_prob, batch_cls)
                cls_loss.append(cls_losses.item())

                # compute regression loss 
                if self.reg_loss_func == 'full':
                    reg_losses = nn.MSELoss()(y_pred_rev, batch_reg)
                if self.reg_loss_func == 'masked':
                    reg_losses = torch.mean(nn.MSELoss(reduction = 'none')(y_pred_rev, batch_reg)*batch_cls)
                reg_loss.append(reg_losses.item())

                # store predicted probability 
                y_pred = y_pred_prob.detach().cpu().numpy().tolist()
                final_output.extend(y_pred)

                del hidden
                del cls_losses
                del reg_losses
                del y_pred

        print("CLS loss: %.4f, REG loss: %.4f"% (np.mean(cls_loss), np.mean(reg_loss)) )
        return np.array(final_output).ravel(), np.mean(cls_loss)+ np.mean(reg_loss), (hiddens, revs)
    
    
class AnomalyDATEModel(nn.Module):
    def __init__(self,max_leaf,importer_size,item_size,dim,head_num=4,fusion_type="concat",act="relu",device="cpu",use_self=True,agg_type="sum", cls_loss_func = 'bce', reg_loss_func = 'full'):
        super(AnomalyDATEModel, self).__init__()
        self.d = dim
        self.device = device
        if act == "relu":
            self.act = nn.LeakyReLU()
        elif act == "mish":
            self.act = Mish() 
        self.fusion_type = fusion_type
        self.use_self = use_self


        # embedding layers 
        self.leaf_embedding = nn.Embedding(max_leaf,dim)
        self.user_embedding = nn.Embedding(importer_size,dim,padding_idx=0)
        self.user_embedding.weight.data[0] = torch.zeros(dim)
        self.item_embedding = nn.Embedding(item_size,dim,padding_idx=0)
        self.item_embedding.weight.data[0] = torch.zeros(dim)

        # attention layer
        self.attention_bolck = Attention(dim,dim,agg_type).to(device)
        self.self_att = MultiHeadAttention(dim,head_num).to(device)
        self.fusion_att = FusionAttention(dim)

        # Hidden & output layer
        self.layer_norm = nn.LayerNorm((100,dim))
        self.fussionlayer = nn.Linear(dim*3,dim)
        self.hidden = nn.Linear(dim,dim)
        self.output_cls_layer = nn.Linear(dim,1)
        self.output_reg_layer = nn.Linear(dim,1)
        
        # Loss function
        self.reg_loss_func = reg_loss_func 
        self.cls_loss_func = cls_loss_func 

    def forward(self,feature,uid,item_id, pretrain=False):
        leaf_vectors = self.leaf_embedding(feature)
        if self.use_self:
            leaf_vectors = self.self_att(leaf_vectors,leaf_vectors,leaf_vectors)
            leaf_vectors = self.layer_norm(leaf_vectors)
        importer_vector = self.user_embedding(uid)
        item_vector = self.item_embedding(item_id)
        query_vector = importer_vector * item_vector
        set_vector, self.attention_w = self.attention_bolck(query_vector,leaf_vectors)
        
        # concat the user, item and tree vectors into a fusion embedding
        if self.fusion_type == "concat":
            fusion = torch.cat((importer_vector, item_vector, set_vector), dim=-1)
            fusion = self.act(self.fussionlayer(fusion))
        elif self.fusion_type == "attention":
            importer_vector, item_vector, set_vector = importer_vector.view(-1,1,self.d), item_vector.view(-1,1,self.d), set_vector.view(-1,1,self.d)
            fusion = torch.cat((importer_vector, item_vector, set_vector), dim=1)
            fusion,_ = self.fusion_att(fusion)
        else:
            raise "Fusion type error"
        hidden = self.hidden(fusion)
        hidden = self.act(hidden)
        
        if pretrain == True:
            classification_output = torch.sigmoid(self.output_cls_layer(hidden))
            regression_output = torch.relu(self.output_reg_layer(hidden))
            return hidden, classification_output, regression_output
        return hidden


    def get_average_hidden_vec(self,train_loader): # calculate average hidden vector on test loader
        with torch.no_grad():
            first = True
            count = 0
            for batch in train_loader:
                batch_feature, batch_user, batch_item, batch_cls, batch_reg = batch
                batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
                batch_feature.to(self.device), batch_user.to(self.device),\
                batch_item.to(self.device), batch_cls.to(self.device), batch_reg.to(self.device)
                batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)
                
                hidden = self.forward(batch_feature,batch_user,batch_item)
                count += 1
                if first:
                    avg_hidden = hidden
                else:
                    avg_hidden += hidden
                    
        avg_hidden /= count
        self.avg_hidden = avg_hidden.mean(dim = 0)
        return self.avg_hidden    
    
    def get_average_hidden_vec_clusters(self,train_loader,n_cluster=20,random_state=0): # calculate average hidden vector on test loader
        X = []
        with torch.no_grad():
            count = 0
            for batch in train_loader:
                batch_feature, batch_user, batch_item, batch_cls, batch_reg = batch
                batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
                batch_feature.to(self.device), batch_user.to(self.device),\
                batch_item.to(self.device), batch_cls.to(self.device), batch_reg.to(self.device)
                batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)
                
                hidden = self.forward(batch_feature,batch_user,batch_item)
                if count == 0:
                    X = hidden
                else:
                    X = torch.cat((X, hidden), dim = 0)                    
                count += 1
                   
        X = X.cpu().numpy()
        kmeans = KMeans(n_clusters=n_cluster, random_state=random_state).fit(X)
        self.avg_hidden = torch.Tensor(kmeans.cluster_centers_).to(self.device)
        return self.avg_hidden
    

  
    def eval_on_batch_for_pretrain(self,test_loader): # predict test data using batch for pretraining
        final_output = []
        cls_loss = []
        reg_loss = []
        hiddens = []
        revs = []  
        i = 0
        
        with torch.no_grad():
            for batch in test_loader:
                i += 1
                batch_feature, batch_user, batch_item, batch_cls, batch_reg = batch
                batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
                batch_feature.to(self.device), batch_user.to(self.device),\
                batch_item.to(self.device), batch_cls.to(self.device), batch_reg.to(self.device)
                batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)
                hidden, y_pred_prob, y_pred_rev = self.forward(batch_feature,batch_user,batch_item, pretrain=True)
                revs.extend(y_pred_rev)
                hiddens.extend(hidden)

                # compute classification loss
                if self.cls_loss_func == 'bce':
                    cls_losses = nn.BCELoss()(y_pred_prob,batch_cls)
                if self.cls_loss_func == 'focal':
                    cls_losses = FocalLoss()(y_pred_prob, batch_cls)
                cls_loss.append(cls_losses.item())

                # compute regression loss 
                if self.reg_loss_func == 'full':
                    reg_losses = nn.MSELoss()(y_pred_rev, batch_reg)
                if self.reg_loss_func == 'masked':
                    reg_losses = torch.mean(nn.MSELoss(reduction = 'none')(y_pred_rev, batch_reg)*batch_cls)
                reg_loss.append(reg_losses.item())

                # store predicted probability 
                y_pred = y_pred_prob.detach().cpu().numpy().tolist()
                final_output.extend(y_pred)

                del hidden
                del cls_losses
                del reg_losses
                del y_pred

        print("CLS loss: %.4f, REG loss: %.4f"% (np.mean(cls_loss), np.mean(reg_loss)) )
        return np.array(final_output).ravel(), np.mean(cls_loss)+ np.mean(reg_loss), (hiddens, revs)


    def eval_on_batch(self,test_loader): # predict test data using batch 
        hiddens = []
        labels = []
        normality_scores = []  
        i = 0
        
        with torch.no_grad():
            for batch in test_loader:
                i += 1
                batch_feature, batch_user, batch_item, batch_cls, batch_reg = batch
                batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
                batch_feature.to(self.device), batch_user.to(self.device),\
                batch_item.to(self.device), batch_cls.to(self.device), batch_reg.to(self.device)
                batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)
                hidden = self.forward(batch_feature,batch_user,batch_item)
                
                if len(self.avg_hidden.shape) == 2:
                    distance_matrix = torch.cdist(hidden, self.avg_hidden)
                    normality_score = distance_matrix.min(dim=1).values
                else: 
                    normality_score = torch.norm(hidden - self.avg_hidden, dim=-1)

                normality_scores.extend(normality_score.cpu().numpy())
                hiddens.extend(hidden)
                labels.extend(batch_cls.cpu().data)
                
            test_auc = roc_auc_score(np.array(labels), np.array(normality_scores))

        print("Test AUC: %.4f"% (test_auc))
        return normality_scores, test_auc, hiddens
