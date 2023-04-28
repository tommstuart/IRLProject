import numpy as np 
import math 
class Policy:
    def __init__(self, q, actions): 
        self.q = q 
        self.actions = actions 
    def __call__(self,s,t):
        raise NotImplementedError
    
def choose_a_from_pi(pi,s,t):
    return np.random.choice([*range(pi.shape[1])], p = pi[s,:,t])

class Boltzmann(Policy): 
    #larger alpha means agent sticks to reward more
    def __init__(self, q, actions, alpha):
        super(Boltzmann, self).__init__(q, actions) 
        
        self.actions = actions
        #Sets up the distribution i.e. self.dist[s,:,t] is a distribution over actions for a given 
        #state-time pair 
        temp = 1/alpha 
        exp_q = np.exp(q/temp) 
        sums = np.sum(exp_q, axis = 1) 
        sums = sums[:,np.newaxis, :] 
        self.dist = exp_q/sums 

        #ln(exp_q/sums) = ln(exp_q) - ln(sums) = q/temp - ln(sums) 
        self.logdist = q/temp - np.log(sums)

    def __call__(self, s, t): 
        return np.random.choice(self.actions, p = self.dist[s,:,t])