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
    def __init__(self, q, actions, alpha = 0.5):
        super(Boltzmann, self).__init__(q, actions) 
        self.temp = 1/alpha 
        self.actions = actions
    #This is normalised wrong but it doesn't make a difference for now, the np.randomchoice figures it all out. 
    def __call__(self, s, t): 
        q_vals = [] 
        for a in self.actions: 
            q_vals.append(self.q[s,a,t]) 
        dist = self.getDistribution(q_vals) 
        return np.random.choice(self.actions, p = dist) 
    
    def getDistribution(self, q_vals):
        exp_q = np.exp(q_vals/self.temp) 
        sums = np.sum(exp_q, axis = 1) 
        sums = sums[:,np.newaxis, :] 
        return exp_q/sums 