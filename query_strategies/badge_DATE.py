import numpy as np
from torch.utils.data import DataLoader
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import deepcopy
import math
import torch
from torch import nn
import torchfile
from torch.autograd import Variable

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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from .strategy import Strategy

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    # print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
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

class DATEBadgeSampling(Strategy):
    def __init__(self, model_path, test_loader, uncertainty_module, args):
        super(DATEBadgeSampling,self).__init__(model_path, test_loader, args)
        self.uncertainty_module = uncertainty_module

    def query(self, k):
        gradEmbedding  = self.get_grad_embedding()
        # normalize
        gradEmbedding = normalize(gradEmbedding, axis = 1, norm = 'l2')
        # get uncertainty
        uncertainty_score = self.get_uncertainty()
        revs = np.asarray(self.get_revenue())
        # integrate revenue and uncertainty
        assert len(gradEmbedding) == len(uncertainty_score)
        for idx in range(len(gradEmbedding)):
            gradEmbedding[idx] = [emb*math.log(2+revs[idx])*uncertainty_score[idx] for emb in gradEmbedding[idx]]
        chosen = init_centers(gradEmbedding, k)
        return self.available_indices[chosen].tolist()

    def get_uncertainty(self):
        uncertainty = self.uncertainty_module.measure(self.uncertainty_module.test_data ,'feature_importance')
        return np.asarray(uncertainty)[self.available_indices]

