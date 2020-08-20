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
from sklearn.cluster import KMeans
from . import utils
from .strategy import Strategy

class DiversitySampling(Strategy):
    def __init__(self, model_path, test_loader, uncertainty_module, args):
        super(DiversitySampling, self).__init__(model_path, test_loader, uncertainty_module, args)

    def query(self, k, beta = 5):
        # get embedding
        emb = self.get_embedding()
        # get uncertainty and revenue
        uncertainty_score = self.get_uncertainty()
        revs = self.get_revenue()
        # get score
        score = [i[0] * math.log(2+i[1]) for i in zip(uncertainty_score, revs)]
        # select beta*k best score
        filtered = np.argpartition(score, -beta*k)[-beta*k:].tolist()
        # get actual data:
        filter_score =  np.asarray(score)[filtered]
        filter_emb = np.asarray(score)[filtered].reshape(-1, 1)
        # prefilter:
        kmeans = KMeans(n_clusters = k, random_state = 42, n_jobs = -1)
        kmeans = kmeans.fit(filter_emb, sample_weight = filter_score)
        cluster = kmeans.predict(filter_emb, sample_weight = filter_score)
        centroids = kmeans.cluster_centers_
        # pick sample nearest to each centroid
        idx = []
        for j in range(len(centroids)):
            dists = kmeans.transform(filter_emb)[:, j]
            idx.append(np.argpartition(dists, 1)[0])
        # map to original
        chosen = [filtered[i] for i in idx]
        return chosen

