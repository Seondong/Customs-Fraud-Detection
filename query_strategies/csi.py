
import numpy as np
import random
import sys
import math
from datetime import datetime, timedelta
import torch
from .DATE import DATESampling
from .drift import DriftSampling
from utils import timer_func
import scipy.io
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.eye_(m.weight)
    elif classname.find('Emb') != -1:
        init.normal(m.weight, mean=0, std=0.01)


class netC1(nn.Module):
    def __init__(self, d, ndf, nc):
        super(netC1, self).__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(d, ndf, kernel_size=1, bias=False),
        )
        self.head = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf, nc, kernel_size=1, bias=True),
        )

    def forward(self, input):
        tc = self.trunk(input)
        ce = self.head(tc)
        return tc, ce


##opt_tc_tabular.py

def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss


class TransClassifierTabular():
    def __init__(self, m, lmbda, batch_size, ndf, n_rots, d_out, eps, n_epoch, lr):
        # self.ds = args.dataset
        self.m = m
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.ndf = ndf
        self.n_rots = n_rots
        self.d_out = d_out
        self.eps = eps
        self.n_epoch = n_epoch

        self.netC = netC1(self.d_out, self.ndf, self.n_rots).cuda()
        weights_init(self.netC)
        self.optimizerC = optim.Adam(self.netC.parameters(), lr, betas=(0.5, 0.999))

    def fit_trans_classifier(self, train_xs, x_test):
        labels = torch.arange(self.n_rots).unsqueeze(0).expand((self.batch_size, self.n_rots)).long().cuda()
        celoss = nn.CrossEntropyLoss()
        # print('Training')
        for epoch in range(self.n_epoch):
            self.netC.train()
            rp = np.random.permutation(len(train_xs))
            n_batch = 0
            sum_zs = torch.zeros((self.ndf, self.n_rots)).cuda()

            for i in range(0, len(train_xs), self.batch_size):
                self.netC.zero_grad()
                batch_range = min(self.batch_size, len(train_xs) - i)
                train_labels = labels
                if batch_range == len(train_xs) - i:
                    train_labels = torch.arange(self.n_rots).unsqueeze(0).expand(
                        (len(train_xs) - i, self.n_rots)).long().cuda()
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(train_xs[rp[idx]]).float().cuda()
                tc_zs, ce_zs = self.netC(xs)
                sum_zs = sum_zs + tc_zs.mean(0)
                tc_zs = tc_zs.permute(0, 2, 1)

                loss_ce = celoss(ce_zs, train_labels)
                er = self.lmbda * tc_loss(tc_zs, self.m) + loss_ce
                er.backward()
                self.optimizerC.step()
                n_batch += 1

            means = sum_zs.t() / n_batch
            means = means.unsqueeze(0)
            self.netC.eval()

            with torch.no_grad():
                val_probs_rots = np.zeros((len(x_test), self.n_rots))
                for i in range(0, len(x_test), self.batch_size):
                    batch_range = min(self.batch_size, len(x_test) - i)
                    idx = np.arange(batch_range) + i
                    xs = torch.from_numpy(x_test[idx]).float().cuda()
                    zs, fs = self.netC(xs)
                    zs = zs.permute(0, 2, 1)
                    diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)

                    diffs_eps = self.eps * torch.ones_like(diffs)
                    diffs = torch.max(diffs, diffs_eps)

                    logp_sz = torch.nn.functional.softmax(diffs, dim=2)
                    val_probs_rots[idx] = torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy()

                val_probs_rots = val_probs_rots.sum(1).mean()

        return val_probs_rots


class CSISampling(DriftSampling):

    def __init__(self, args):
        super(CSISampling, self).__init__(args)
        assert len(self.subsamps) == 2

    # trainn_add_tabular

    def load_trans_data(self, n_rots, d_out):
        valid_embeddings, test_embeddings = self.generate_DATE_embeddings()
        train_real = valid_embeddings.cpu().numpy()
        test_embeddings = test_embeddings.cpu().numpy()
        ind = len(test_embeddings) // 2
        val_real = test_embeddings[:ind]
        val_fake = test_embeddings[ind:]

        n_train, n_dims = train_real.shape
        rots = np.random.randn(n_rots, n_dims, d_out)

        # print('Calculating transforms')
        x_train = np.stack([train_real.dot(rot) for rot in rots], 2)
        val_real_xs = np.stack([val_real.dot(rot) for rot in rots], 2)
        val_fake_xs = np.stack([val_fake.dot(rot) for rot in rots], 2)
        x_test = np.concatenate([val_real_xs, val_fake_xs])
        return x_train, x_test

    def train_anomaly_detector(self, m, lmbda, batch_size, ndf, n_rots, d_out, eps, n_epoch, lr):
        x_train, x_test = self.load_trans_data(n_rots, d_out)
        tc_obj = TransClassifierTabular(m, lmbda, batch_size, ndf, n_rots, d_out, eps, n_epoch, lr)
        probs = tc_obj.fit_trans_classifier(x_train, x_test)
        return probs

    def concept_drift(self):
        lr = 0.001
        n_rots = 256
        batch_size = 64
        n_epoch = 1
        d_out = 32
        dataset = 'custom'
        exp = 'affine'
        c_pr = 0
        true_label = 1
        ndf = 8
        m = 1
        lmbda = 0.1
        eps = 0
        n_iters = 1

        score = self.train_anomaly_detector(m, lmbda, batch_size, ndf, n_rots, d_out, eps, n_epoch, lr)
        score = score*25
        if score > 1: score = 1
        print("Anomaly score", score)
        return score

    def query(self, k):
        # Drift sampler should measure the concept drift and update subsampler weights before the query selection is made.
        self.update_subsampler_weights()
        super(CSISampling, self).query(k)
        return self.chosen