import numpy as np 
import math 
class Policy:
    def __init__(self, q, actions): 
        self.q = q 
        self.actions = actions 
    def __call__(self,s, t): 
        raise NotImplementedError 
    
def choose_a_from_pi(pi,s,t):
    return np.random.choice([*range(pi.shape[1])], p = pi[s,:,t])

class Boltzmann(Policy): 
    def __init__(self, q, alpha = 0.1):
        super(Boltzmann, self).__init__(q) 
        self.temp = 1/alpha 

        def __call(self, s, t): 
            q_vals = [] 
            for a in self.actions: #find the q value for each action 
                q_vals.append(self.q[s,a,t]) 
            exp_q = math.exp(q_vals/self.temperature) 
            z = np.sum(exp_q) 
            return np.random.choice([*range(self.q.shape[-1])], p = exp_q/z) #make the distribution and pick a random action according to the distn
        
        def getDistribution(self, s, t):
            q_vals = [] 
            for a in self.actions: 
                q_vals.append(self.q[s,a,t])
            exp_q = math.exp(q_vals/self.temperature) 
            z = np.sum(exp_q) 
            return exp_q/z 