import numpy as np
import math
from utils import timer_func

class ExpWeights(object):
    """ Expenential weight helper, adapted form RP1 paper 
        sample: sample the weights based on its underlying distribution l
        update_dists(feedback): update the underlying distribution with new feedback (should be loss)
    """
    def __init__(self, 
                 num = 21,
                 lr = 2,
                 window = None, # we don't use this yet.. 
                 epsilon = 0,
                 decay = 1,
                 alpha = 0):
        
        self.num = num
        self.arms = [i/(num-1) for i in range(num)]
        self.l = np.array([1.0 for i in range(len(self.arms))])
        self.unfiltered_p = np.array([1/num for x in range(len(self.arms))])
        self.filter = np.array([1]*self.num)
        self.p = np.array([1/num for x in range(len(self.arms))])
        self.arm = 0
        self.value = self.arms[self.arm]
        self.reward_buffer = []
        self.count = []
        self.window = window
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay
        self.choices = [self.arm]
        self.feedbacks = []
        self.alpha = alpha
        
    def reinit(self):
        num = self.num
        self.p = np.array([1/num for x in range(len(self.arms))])
        self.filter = np.array([1]*self.num)
        self.unfiltered_p = np.array([1/num for x in range(len(self.arms))])
        self.l = np.array([1.0 for i in range(len(self.arms))])
        self.reward_buffer = []

        
    def set_data(self, data):
        self.data = data
        
        
    def sample(self):
        self.unfiltered_p = self.l / np.sum(self.l) # normalize to make it a distribution
#             print(f'p = {self.p}')
        self.unfiltered_p = self.unfiltered_p*(1 - self.epsilon) + self.epsilon/self.num
        self.p = self.unfiltered_p*self.filter
        self.p /= np.sum(self.p)
        self.arm = np.random.choice(range(0, self.num), p=self.p)

        self.value = self.arms[self.arm]
        self.choices.append(self.arm)
        
        return(self.value)
        
        
    def update_dists(self, feedback, norm=1):
        
        # Need to normalize score. 
        # Since this is non-stationary, subtract mean of previous 5. 
        if not math.isfinite(feedback):
            return
        print(f'Performance: {feedback}')
        
        if self.window:
            self.reward_buffer.append(feedback)
            self.reward_buffer = self.reward_buffer[-self.window :]
            mean = np.mean(self.reward_buffer)
        else:
            self.reward_buffer = [error*self.decay for error in self.reward_buffer]
            self.count = [c*self.decay for c in self.count]
            self.reward_buffer.append(feedback)        
            self.count.append(1)
            # if feedback is positive, the current setting produced better results than the recent five trials.
            mean = np.sum(self.reward_buffer)/np.sum(self.count)

        print(f'Buffer (last 5): {self.reward_buffer[-5:]}')
        print(f'Buffer (mean): {mean}')

        feedback = (feedback-mean)/feedback
        # feedback /= norm
        
        print(f'Feedback: {feedback}')
        
        # Thus, the probability of pulling the arm is reduced.
        # self.l[self.arm] *= self.decay
        mean_l = np.mean(self.l)
        self.l[self.arm] *= np.exp(self.lr * feedback/self.unfiltered_p[self.arm])
        
        self.l += self.alpha*np.e*mean_l

        max_l = np.max(self.l)
        if max_l > 1e10:
            self.l *= 1e10/max_l  

        self.feedbacks.append(feedback)
        
        
    def update_dists_advanced(self, each_chosen, feedback, norm=1):     
        # Need to normalize score. 
        # Since this is non-stationary, subtract mean of previous 5. 
        if not math.isfinite(feedback):
            return
        self.reward_buffer.append(feedback)
        self.reward_buffer = self.reward_buffer[-5:]
        
        # if feedback is positive, the current setting produced worse results than the recent five trials.
        feedback -= np.mean(self.reward_buffer)
        feedback /= norm
        
        print(f'Feedback: {feedback}')
    
        self.l[self.arm] *= self.decay
        self.l[self.arm] -= self.lr * feedback/max(self.p[self.arm], 1e-16)
        
        self.feedbacks.append(feedback)
        
        
class UCB(object):
    def __init__(self, num, alpha = 0.3, gamma = 0.9, window = None):
        self.dc = 1
        self.gamma = gamma
        self.alpha = alpha
        self.window = window
        
        self.num = num
        self.arms = [i/(num-1) for i in range(num)]
        
        # Track the number of times we pull each arm
        self.counts = np.array([0.0]*self.num)

        # Track the current average reward of the arm
        self.unfiltered_p = np.array([0.0]*self.num)
        self.filter = np.array([1]*self.num)
        self.p = self.unfiltered_p*self.filter

        # Track 
        self.r = np.array([0.0]*self.num)

        # Current arm and value
        self.arm = 0
        self.value = 0

        # When should we switch the arm?
        self.__next_update = 0

    def reinit(self):
        # Track the number of times we pull each arm
        self.counts = np.array([0.0]*self.num)

        # Track the current average reward of the arm
        self.unfiltered_p = np.array([0.0]*self.num)
        self.filter = np.array([1]*self.num)
        self.p = self.unfiltered_p*self.filter

        # Track 
        self.r = np.array([0.0]*self.num)
        self.dc = 1


    def set_data(self, data):
        self.data = data 

    def __tau(self, r):
        return int(math.ceil((1 + self.alpha) ** r))

    def __bonus(self, n, r):
        tau = self.__tau(r)
        bonus = math.sqrt((1. + self.alpha) * math.log(math.e * float(n) / tau) / (2 * tau))
        return bonus
    def __set_arm(self, arm):
        """
        When choosing a new arm, make sure we play that arm for
        tau(r+1) - tau(r) episodes.
        """
        self.arm = arm
        self.__next_update += max(1, self.__tau(self.r[arm] + 1) - self.__tau(self.r[arm]))
        self.r[arm] += 1
        self.value = self.arms[arm]

    def sample(self):
        # Play each arm once
        for arm in range(self.num):
            if self.counts[arm] == 0:
                self.arm = arm
                self.value = self.arms[arm]
                return self.value
        
        if not self.window:
            self.dc = self.gamma
        else:
            pass
            # self.counts = self.counts[-self.window :]
            # self.unfiltered_p = self.unfiltered_p[-self.window :]
            # self.p = self.p[-self.window :]
            # self.r = self.r[-self.window :]
        # Before next update, keep the arm
        if self.__next_update > sum(self.counts):
            return self.value

        # Updation
        # Calculate the estimated reward to maximize
        ucb_values = np.array([0.0]*self.num)
        total_counts = np.sum(self.counts)
        for arm in range(self.num):
            bonus = self.__bonus(total_counts, self.r[arm])
            ucb_values[arm] = self.p[arm] + bonus

        # Pull the max arm
        chosen_arm = np.argmax()
        self.__set_arm(chosen_arm)
        return self.value
    
    def update_dists(self, feedback):
        # Decay
        self.counts *= self.dc
        self.unfiltered_p *= self.dc
        self.r *= self.dc
        
        # Get number of time the arm is chosen
        self.counts[self.arm] = self.counts[self.arm] + 1
        n = self.counts[self.arm]

        # Update
        current_reward = self.unfiltered_p[self.arm]
        new_reward = ((n - 1) / float(n)) * current_reward + (1 / float(n)) * feedback
        self.unfiltered_p[self.arm] = new_reward
        self.p = self.unfiltered_p*self.filter