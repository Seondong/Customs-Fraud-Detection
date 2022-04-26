import numpy as np
import random
import sys
import math
from .DATE import DATESampling
from .weight_sampler import ExpWeights, UCB
from .hybrid import HybridSampling
from utils import timer_func
        
class AdaHybridSampling(HybridSampling):
    """Adaptive Performance Tuning (APT) Strategy" - Finding the best exploration ratio by using the performance signal. Currently supports two strategies, preferably in the order of exploitation/exploration. The description of this strategy is introduced in Sec 4.1. of our ICDMW 2021 paper (https://arxiv.org/pdf/2109.14155.pdf) """
    
    def __init__(self, args):
        super(AdaHybridSampling,self).__init__(args)
        assert len(self.subsamps) == 2   # TODO: Ideally, it should support multiple strategies
        if args.ada_algo == 'exp3':
            if args.ada_discount != 'window':
                wd = None
            else:
                wd = 5
            print(wd)
            self.weight_sampler = ExpWeights(lr = self.args.ada_lr, num = args.num_arms, epsilon = args.ada_epsilon, decay = args.ada_decay, window = wd) # initialize it at the beginning of the simulation

        if args.ada_algo == 'exp3s':
            if args.ada_discount != 'window':
                wd = None
            else:
                wd = 5
            print(wd)
            self.weight_sampler = ExpWeights(lr = self.args.ada_lr, num = args.num_arms, epsilon = args.ada_epsilon, decay = args.ada_decay, window = wd, alpha = 0.001) # initialize it at the beginning of the simulation

        if args.ada_algo == 'ucb':
            if args.ada_discount != 'window':
                wd = None
            else:
                wd = 25
            print(wd)
            self.weight_sampler = UCB(num = args.num_arms, gamma = args.ada_decay, window = wd) # initialize it at the beginning of the simulation
   
   
    def update_subsampler_weights(self, performance):
        # Update weights for next week
        self.weight_sampler.set_data(self.data)
        self.weight = self.weight_sampler.sample()
        self.weights = [1 - self.weight, self.weight]
        print(f'weight_sampler.p = {self.weight_sampler.p}')
        
        # Update underlying distribution for each arm using predicted results
        self.weight_sampler.update_dists(performance)
        # self.weight_sampler.update_dists_advanced(self.each_chosen, 1-performance)
        print(f'Ada arm: {self.weight_sampler.value}')
        try:
            print(f'Ada distribution: {self.weight_sampler.l}')
        except:
            pass
        print(f'Reward (accuracy): {performance}')        
#         logger.info(f'Ada distribution: {self.weight_sampler.p}')
#         logger.info(f'Ada arm: {self.weight_sampler.value}')
#         logger.info(f'Feedbacks: {self.weight_sampler.data}')