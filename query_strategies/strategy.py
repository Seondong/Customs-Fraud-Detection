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

from . import utils

class Strategy:
    def __init__(self, model_path, test_loader, uncertainty_module, args):
        self.test_loader = test_loader
        self.dim = args.dim
        self.model_path = model_path
        self.device = args.device
        self.uncertainty_module = uncertainty_module

    def query(self, k):
        pass

    def get_model(self):
        return torch.load(self.model_path)

    def get_output(self):
        best_model = self.get_model()
        final_output, _, (hiddens, revs) = best_model.module.eval_on_batch(self.test_loader)
        return final_output

    def get_revenue(self):
        best_model = self.get_model()
        final_output, _, (hiddens, revs) = best_model.module.eval_on_batch(self.test_loader)
        return revs

    def get_uncertainty(self):
        """
        with open("./processed_data.pickle","rb") as f :
            processed_data = pickle.load(f)

        train = processed_data["raw"]["train"]
        valid = processed_data["raw"]["valid"]
        test = processed_data["raw"]["test"]

        uncertainty_score = np.asarray(utils.uncertainty_measurement(train, valid, test, 'feature_importance'))
        """
        return self.uncertainty_module.measure(self.uncertainty_module.test_data ,'feature_importance')

    def get_embedding(self):
        best_model = self.get_model()
        final_output, _, (hiddens, revs) = best_model.module.eval_on_batch(self.test_loader)
        return hiddens

    def get_grad_embedding(self):
        embDim = self.dim
        best_model = torch.load(self.model_path)
        final_output, _, (hiddens, revs) = best_model.module.eval_on_batch(self.test_loader)
        num_data = self.test_loader.dataset.tensors[-1].shape[0]
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
                if self.device == 'cpu':
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
            return embedding
