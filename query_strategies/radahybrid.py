import numpy as np
import random
import sys
import math
from .DATE import DATESampling
from .drift import DriftSampling
from .adahybrid import AdaHybridSampling
from main import initialize_sampler


import ot
from datetime import datetime, timedelta

import pandas as pd
import torch

        
class RegulatedAdaHybridSampling(AdaHybridSampling):

    def __init__(self, args):
        super(RegulatedAdaHybridSampling,self).__init__(args)   
        self.drift_detector = initialize_sampler(args.drift, args)

    def set_data(self, data):
        super(RegulatedAdaHybridSampling, self).set_data(data)
        self.drift_detector.set_data(data)

    def update_subsampler_weights(self, performance):

        self.weight_sampler.set_data(self.data)
        self.weight = self.weight_sampler.sample()
        self.weights = [self.weight, 1 - self.weight]
        print(f'weight_sampler.p = {self.weight_sampler.p}')
        self.dms_weight = round(self.drift_detector.concept_drift(), 2)
        
        if self.args.mixing == 'multiply':
            updated_performance = performance * (1 - self.dms_weight)
            # Update underlying distribution for each arm using predicted results
            self.weight_sampler.update_dists(1-updated_performance)
            # self.weight_sampler.update_dists_advanced(self.each_chosen, 1-performance)

        if self.args.mixing == 'reinit':
            if self.dms_weight > 0.5:
                self.weight_sampler.reinit()
            else: 
                self.weight_sampler.update_dists(1-performance)

        print(f'Ada distribution: {self.weight_sampler.p}')
        print(f'Ada arm: {self.weight_sampler.value}')
        print(f'Feedbacks: {self.weight_sampler.data}')        

    def query(self, k):
        self.drift_detector.update_subsampler_weights()
        super(RegulatedAdaHybridSampling, self).query(k)
        return self.chosen