import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances
from .DATE import DATESampling


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = {ind}
    centInds = [0.] * len(X)
    cent = 0
    # print('#Samps\tTotal Distance')
    while len(mu) < K:             
        # Sundong: ideally it should be changed from len(mu) to len(indsAll), but the sampling would not terminate.     
        # While debugging in 100% inspection scenario, it selects duplicated items, resulting in smaller number of inspections than random selection
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.add(ind)
        cent += 1
    indsAll = list(indsAll)
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    
    # Added by Sundong: 
    # Handling edge cases: if the number of selected items is less than expected, select remaining portion as random. 
    unselected = np.setdiff1d(np.arange(len(D2)), indsAll)
    additionally_selected_inds = np.random.choice(unselected, K-len(indsAll), replace=False)
    indsAll.extend(additionally_selected_inds)
    
    assert K == len(indsAll)
    return indsAll


class BadgeSampling(DATESampling):
    """ BADGE strategy: BADGE model uses the embeddings of the base model (DATE) and find the most diverse imports by KMeans++.
        Reference: Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds; ICLR 2020 """
    
    def __init__(self, args):
        super(BadgeSampling,self).__init__(args)

    def query(self, k, model_available = False):
        # Train DATE model
        if not model_available:
            self.train_xgb_model()
            self.prepare_DATE_input()
            self.train_DATE_model()
        
        # Get embeddings from DATE, run BADGE strategy
        gradEmbedding  = self.get_grad_embedding()
        chosen = init_centers(gradEmbedding, k)
        return self.available_indices[chosen].tolist()