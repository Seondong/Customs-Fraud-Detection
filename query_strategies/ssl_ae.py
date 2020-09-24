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


class SSLAutoencoderSampling(Strategy):
    
    
    def __init__(self, model_path, test_data, test_loader, args):
        self.model_path = './intermediary/xgb_model-'+args.identifier+'.json'
        self.identifier = args.identifier
        super(sslSampling,self).__init__(model_path, test_data, test_loader, args)
    
    
    def get_autoencoder_model(self):
        xgb_clf = XGBClassifier(n_estimators=100, max_depth=4,n_jobs=-1)
        xgb_clf.load_model(self.model_path)
        return xgb_clf
    
    
    def load_test_data(self):
        _, _, _, _, _, _,_, _, _, _, _, _, _, _, xgb_testx, _ = separate_train_test_data(self.identifier)
        return xgb_testx
    
    
    def get_ssl_output(self):
        xgb_clf = self.get_autoencoder_model()
        xgb_testx = self.load_test_data()
        final_output = xgb_clf.predict_proba(xgb_testx)[:,1]
        return final_output[self.available_indices]

    
    def query(self, k):
        output = self.get_ssl_output()
        chosen = np.argpartition(output, -k)[-k:]
        return self.available_indices[chosen].tolist()
   

