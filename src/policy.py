import numpy as np 
import math 
class Policy:
    def __init__(self, q, actions): 
        self.q = q 
        self.actions = actions 
    def __call__(self,s, t): 
        raise NotImplementedError 

class Boltzmann(Policy): 
    def __init__(self, q, alpha = 0.1):
        super(Boltzmann, self).__init__(q) 
        self.temp = 1/alpha 

        def __call(self, s, t): 
            q_vals = [] 
            for action in self.actions: #find the q value for each action 
                q_vals.append(self.q(s, action, t)) 
                q_vals.append(self.q(s, action, t)) 
            exp_q = math.exp(q_vals/self.temperature) 
            z = np.sum(exp_q) 
            return np.random.choice([*range(self.q.shape[-1])], p = exp_q/z) #make the distribution and pick a random action according to the distn