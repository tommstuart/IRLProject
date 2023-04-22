import math
def Phi(theta, t, x): 
    return math.exp(-(x-theta(t)) ** 2)

def sawtooth(t, k):
    return 2*k*abs(t/(2*k) - math.floor(t/(2*k) + 0.5))

class SingleStateReward:
    def __init__(self, R_max, n_actions):
        self.R_max = R_max  
        self.n_actions = n_actions 
    def __call__(self, s, a, t): 
        return self.R_max*Phi(lambda t: sawtooth(t,self.n_actions), t, a)
