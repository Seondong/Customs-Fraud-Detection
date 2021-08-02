import numpy as np
import torch
from .DATE import DATESampling
from .hybrid import HybridSampling

            
class DriftSampling(HybridSampling):
    def __init__(self, args):
        super(DriftSampling,self).__init__(args)
        assert len(self.subsamps) == 2
        # self.data already exists - In main.py, we declared in: sampler.set_data(data)
 
    def generate_DATE_embeddings(self):
        date_sampler = DATESampling(self.args)
        date_sampler.set_data(self.data)
        date_sampler.train_xgb_model()
        date_sampler.prepare_DATE_input()
        date_sampler.train_DATE_model()
        valid_embeddings = torch.stack(date_sampler.get_embedding_valid())  # Embeddings for validation data
        test_embeddings = torch.stack(date_sampler.get_embedding_test())         # Embeddings for test data

        return valid_embeddings, test_embeddings


    def domain_shift(self):
        pass


    def update_subsampler_weights(self):  
        self.dms_weight = round(self.domain_shift(), 2)
        self.weights = [1 - self.dms_weight, self.dms_weight]



    