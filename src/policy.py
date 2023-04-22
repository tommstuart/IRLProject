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
    def __init__(self, q, actions, alpha = 0.5): #larger alpha means agent sticks to reward more
        super(Boltzmann, self).__init__(q, actions) 
        self.temp = 1/alpha 
        self.actions = actions

    def __call__(self, s, t): #This is normalised wrong but it doesn't make a difference for now, the np.randomchoice figures it all out. 
        q_vals = [] 
        for a in self.actions: #find the q value for each action 
            q_vals.append(self.q[s,a,t]) 
        exp_q = np.exp(np.asarray(q_vals)/self.temp) 
        z = np.sum(exp_q) 
        return np.random.choice(self.actions, p = exp_q/z)
    
    def getDistribution(self, q_vals):
        exp_q = np.exp(q_vals/self.temp) 
        sums = np.sum(exp_q, axis = 1) 
        sums = sums[:,np.newaxis, :] 
        return exp_q/sums 