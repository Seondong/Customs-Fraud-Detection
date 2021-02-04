import numpy as np
import random
import sys
import math
from .DATE import DATESampling
from .strategy import Strategy
from .hybrid import HybridSampling


class ExpWeights(object):
    """ Expenential weight helper, adapted form RP1 paper 
        sample: sample the weights based on its underlying distribution l
        update_dists(feedback): update the underlying distribution with new feedback (should be loss)
    """
    def __init__(self, 
                 arms=[0.99,0.98,0.95,0.9,0.8,0.7,0.6,0.5],
                 lr = 2,
                 window = 20, # we don't use this yet.. 
                 epsilon = 0,
                 decay = 0.95):
        
        self.arms = arms
        self.l = {i:0 for i in range(len(self.arms))}
        self.arm = 0
        self.value = self.arms[self.arm]
        self.error_buffer = []
        self.window = window
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay
        self.choices = [self.arm]
        self.feedbacks = []
    
    
    def set_data(self, data):
        self.data = data
        
        
    def sample(self):
        if np.random.uniform() > self.epsilon:
            self.p = [np.exp(x) for x in self.l.values()]
            self.p /= np.sum(self.p) # normalize to make it a distribution
#             print(f'p = {self.p}')
            self.arm = np.random.choice(range(0,len(self.p)), p=self.p)
        else:
            self.arm = int(np.random.uniform() * len(self.arms))

        self.value = self.arms[self.arm]
        self.choices.append(self.arm)
        
        return(self.value)
        
        
    def update_dists(self, feedback, norm=1):
        
        # Need to normalize score. 
        # Since this is non-stationary, subtract mean of previous 5. 
        if not math.isfinite(feedback):
            return
        self.error_buffer.append(feedback)
        self.error_buffer = self.error_buffer[-5:]
        
        # if feedback is positive, the current setting produced worse results than the recent five trials.
        feedback -= np.mean(self.error_buffer)
        feedback /= norm
        
        print(f'Feedback: {feedback}')
        
        # Thus, the probability of pulling the arm is reduced.
        self.l[self.arm] *= self.decay
        self.l[self.arm] -= self.lr * feedback/max(self.p[self.arm], 1e-16)
        
        self.feedbacks.append(feedback)
        
        
    def update_dists_advanced(self, each_chosen, feedback, norm=1):
        # Calculate feedback based on inspected_imports
        
        # each_chosen을 활용하여 PERFORMANCE을 계산하면 되는데, 그 전에 subset을 자를 수 있다. 
#         import pdb
#         pdb.set_trace()
#         each_chosen
        
         # Indices of sampled imports (Considered as fraud by model) -> This will be inspected thus annotated.    
#         indices = [point + data.offset for point in chosen]
        
#         inspected_imports = data.df.iloc[indices]
        
        
        
        # Need to normalize score. 
        # Since this is non-stationary, subtract mean of previous 5. 
        if not math.isfinite(feedback):
            return
        self.error_buffer.append(feedback)
        self.error_buffer = self.error_buffer[-5:]
        
        # if feedback is positive, the current setting produced worse results than the recent five trials.
        feedback -= np.mean(self.error_buffer)
        feedback /= norm
        
        print(f'Feedback: {feedback}')
    
        self.l[self.arm] *= self.decay
        self.l[self.arm] -= self.lr * feedback/max(self.p[self.arm], 1e-16)
        
        self.feedbacks.append(feedback)
        
        
class AdaHybridSampling(HybridSampling):
    """ AdaHybrid strategy: Hybrid strategy with Exponential Weight Decay. """
    def __init__(self, args):
        super(AdaHybridSampling,self).__init__(args)
        assert len(self.subsamps) == 2   # TODO: Ideally, it should support multiple strategies
        self.weight_sampler = ExpWeights(lr = self.args.ada_lr) # initialize it at the beginning of the simulation
   
    def update(self, performance):
        # Update weights for next week
        self.weight_sampler.set_data(self.data)
        self.weight = self.weight_sampler.sample()
        self.weights = [self.weight, 1 - self.weight]
        print(f'weight_sampler.p = {self.weight_sampler.p}')
        
        # Update underlying distribution for each arm using predicted results
#         self.weight_sampler.update_dists(1-performance)
        self.weight_sampler.update_dists_advanced(self.each_chosen, 1-performance)
        print(f'Ada distribution: {self.weight_sampler.p}')
        print(f'Ada arm: {self.weight_sampler.value}')
        print(f'Feedbacks: {self.weight_sampler.data}')        
#         logger.info(f'Ada distribution: {self.weight_sampler.p}')
#         logger.info(f'Ada arm: {self.weight_sampler.value}')
#         logger.info(f'Feedbacks: {self.weight_sampler.data}')