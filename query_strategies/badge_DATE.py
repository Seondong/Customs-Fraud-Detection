import numpy as np
from torch.utils.data import DataLoader
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import deepcopy

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

# Get uncertainty

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

class DATEBadgeSampling:
    def __init__(self, model_path, test_loader, args):
        self.test_loader = test_loader
        self.dim = args.dim
        self.model_path = model_path

    def query(self, k):
        gradEmbedding  = self.get_grad_embedding(self.model_path, self.dim, self.test_loader)
        chosen = init_centers(gradEmbedding, k)
        return chosen

    def get_grad_embedding(self, model_path, dim, test_loader):
        embDim = dim
        best_model = torch.load(model_path)
        final_output, _, (hiddens, revs) = best_model.module.eval_on_batch(test_loader)
        num_data = test_loader.dataset.tensors[-1].shape[0]
        nLab = 2
        print(len(final_output), hiddens[0].shape, len(hiddens))
        embedding = np.zeros([num_data, embDim * nLab])
        with torch.no_grad():
            for idx, prob in enumerate(final_output):
                maxInds = np.asarray([0, 0])
                probs = np.asarray([1 - prob, prob])
                if prob >= 0.5:
                    maxInd = 1
                else:
                    maxInd = 0
                for c in range(nLab):
                    if c == maxInd:
                        embedding[idx][embDim * c : embDim * (c+1)] = hiddens[idx] * (1 - probs[c])
                    else:
                        embedding[idx][embDim * c : embDim * (c+1)] = hiddens[idx] * (0 - probs[c])
            # Normalize:
            embedding = normalize(embedding, axis = 1, norm = 'l2')
            # Integrate revenue and uncertainty:
                
            return embedding