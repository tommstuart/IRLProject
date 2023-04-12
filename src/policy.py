import numpy as np 
class Policy:
    def __init__(self, q_values): 
        self.q_values = q_values 
    def __call__(self,s): 
        raise NotImplementedError 

class Boltzmann(Policy): 
    def __init__(self, q_values, alpha = 0.1):
        super(Boltzmann, self).__init__(q_values) 
    
    def __call__(self, s): 
        exp_q = np.exp(self.q_values[s]/self.temperature)
        z = np.sum(exp_q) 
        return np.random.choice([*range(self.q_values.shape[-1])], p=exp_q/z)