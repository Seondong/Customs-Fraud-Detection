import numpy as np
import sys
import gc
from sklearn.cluster import KMeans
from .DATE import DATESampling


class DiversitySampling(DATESampling):
    """ Diversity strategy: Using DATE embedding, then select centroid by KMeans.
        In this way, we can guarantee diverse imports for next inspection. """
    
    def __init__(self, args):
        super(DiversitySampling, self).__init__(args)


    def get_uncertainty(self):
        if self.uncertainty_module is None :
            # return np.asarray(self.get_output().apply(lambda x : -1.8*abs(x-0.5) + 1))
            return np.asarray(-1.8*abs(self.get_output()-0.5) + 1)
        uncertainty = self.uncertainty_module.measure(self.uncertainty_module.test_data ,'feature_importance')
        return np.asarray(uncertainty)[self.available_indices]
    
    
    def diversity_sampling(self, k, beta=5):
        # Get embedding from DATE, run diversity strategy
        emb = self.get_embedding()
        # get uncertainty and revenue
        uncertainty_score = self.get_uncertainty()
        revs = self.get_revenue()
        # get score
        score = [i[0] * self.rev_score()(i[1]) for i in zip(uncertainty_score, revs)]
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
    
    
    def query(self, k, model_available = False):
        if not model_available:
            self.train_xgb_model()
            self.prepare_DATE_input()
            self.train_DATE_model()
        chosen = self.diversity_sampling(k)
        return self.available_indices[chosen].tolist()
        
        
        
