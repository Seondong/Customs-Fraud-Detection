import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from .DATE import DATESampling
from .badge import init_centers


class bATESampling(DATESampling):
    """ bATE strategy: Our proposed model for better exploration. By following the BADGE model, we first uses the embeddings of the base model, DATE. Our contribution is to amplify the embedding with extra uncertainty score, and predicted revenue. Finally, we find the most diverse imports by KMeans++. """
        
      
    def __init__(self, args):
        super(bATESampling, self).__init__(args)
    
    
    def get_uncertainty(self):
        if self.uncertainty_module is None :
            # return np.asarray(self.get_output().apply(lambda x : -1.8*abs(x-0.5) + 1))
            return np.asarray(-1.8*abs(self.get_output()-0.5) + 1)
        uncertainty = self.uncertainty_module.measure(self.uncertainty_module.test_data ,'feature_importance')
        return np.asarray(uncertainty)[self.available_indices]
    
    
    def bATE_sampling(self, k):
        gradEmbedding  = self.get_grad_embedding()
        # normalize
        # gradEmbedding = normalize(gradEmbedding, axis = 1, norm = 'l2')
        # get uncertainty
        uncertainty_score = self.get_uncertainty()
        revs = np.asarray(self.get_revenue())
        # integrate revenue and uncertainty
        assert len(gradEmbedding) == len(uncertainty_score)
        for idx in range(len(gradEmbedding)):
            gradEmbedding[idx] = [emb*self.rev_score()(revs[idx])*uncertainty_score[idx] for emb in gradEmbedding[idx]]
        chosen = init_centers(gradEmbedding, k)
        return chosen
    
    
    def query(self, k, model_available = False):
        if not model_available:
            self.train_xgb_model()
            self.prepare_DATE_input()
            self.train_DATE_model()
        chosen = self.bATE_sampling(k)
        return self.available_indices[chosen].tolist()

    
