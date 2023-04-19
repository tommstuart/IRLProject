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
    def __init__(self, q, actions, alpha = 1.5): #ok so upping the alpha doesn't make a difference. Higher alpha means we're more confident in the expert to make high value decisions 
        #which makes me think that it's going to be something wrong with the V/Q calculations 

        #It does seem to effect the spread of the results though like higher alpha means you get 0.5,..,0.5 ish rewards and then 0 rewards for a while whereas before it was like 0.50.5000.50.5 like fast alternations 
        super(Boltzmann, self).__init__(q, actions) 
        self.temp = 1/alpha 
        self.actions = actions

    def __call__(self, s, t): 
        q_vals = [] 
        for a in self.actions: #find the q value for each action 
            q_vals.append(self.q[s,a,t]) 
        exp_q = np.exp(np.asarray(q_vals)/self.temp) 
        z = np.sum(exp_q) 
        return np.random.choice(self.actions, p = exp_q/z)
    
    def getDistribution(self, q_vals):
        exp_q = np.exp(q_vals/self.temp) 
        z = np.sum(exp_q) # this might need to be along some axes 
        return exp_q/z 