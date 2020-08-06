import numpy as np
from torch.utils.data import DataLoader
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch import nn
import torchfile
from torch.autograd import Variable
import resnet
import vgg
import torch.optim as optim
import pdb
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import time
import numpy as np
import scipy.sparse as sp
from itertools import product
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.externals.six import string_types
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

# ADD uncertainty
# preprocess data: add 1 more column
# def preprocess(df: pd.DataFrame) -> pd.DataFrame:
column_to_use = ['sgd.date','office.id','importer.id', 
                 'declarant.id','tariff.code','country',
                 'cif.value','quantity','gross.weight','fob.value',
                 'total.taxes','revenue','illicit']
def uncertainty(x):
    if x < 0.05:
        return 1
    elif x > 0.6:
        return 0
    return 0.5
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, 'tariff.ratio'] = df['total.taxes'] / df['fob.value']
    df.loc[:,'uncertain'] = df['tariff.ratio'].apply(uncertainty)

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll

class BadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, args):
        super(BadgeSampling, self).__init__(X, Y, idxs_lb, args)

    def query(self, n):
    	# n_pool = len(y_train)
    	# in list of data need to be label, get the unlabel index
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
        # n: number of query point
        chosen = init_centers(gradEmbedding, n),
        return idxs_unlabeled[chosen]

def get_grad_embedding(X,Y,model_path,dim,test_loader):
    # model = self.clf
    embDim = model.get_embedding_dim()
    # model.eval()
    nLab = len(np.unique(Y))
    embedding = np.zeros([len(Y), embDim * nLab])
    # loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
    #                     shuffle=False, **self.args['loader_te_args'])
    with torch.no_grad():
        best_model = torch.load(model_path)
        _, _, batchProbs = best_model.module.eval_on_batch(test_loader)
        maxInds = np.argmax(batchProbs,1)
        for j in range(len(y)):
            for c in range(nLab):
                if c == maxInds[j]:
                    embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                else:
                    embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (0 - 1 * batchProbs[j][c])
        return torch.Tensor(embedding)


