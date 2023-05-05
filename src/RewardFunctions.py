import numpy as np 
def Phi(theta, t, x, sigma): 
    return np.exp(-0.5*(((x-theta(t))/sigma) ** 2))

def sawtooth(t, k):
    return 2*k*abs(t/(2*k) - np.floor(t/(2*k) + 0.5))

def greed(t, c):
    return 1/(t/c + 1) 

class SingleStateReward:
    def __init__(self, R_max, n_actions, sigma = 1):
        self.R_max = R_max  
        self.n_actions = n_actions 
        self.sigma = sigma 
    def __call__(self, s, a, t): 
        return self.R_max*Phi(lambda t: sawtooth(t,self.n_actions-1), t, a, self.sigma)

class DoubleStateReward:
    def __init__(self, R_max, n_actions, greed_longevity, sigma = 1 ): 
        self.R_max = R_max 
        self.n_actions = n_actions 
        self.c = greed_longevity
        self.sigma = sigma 
    #Going to say 0 is not hungry and 1 is hungry 
    #Action 0 will be doing nothing and then action j will be buying j - so I might want to shift my phi 
    def __call__(self, s, a, t): 
        if s==0: #not hungry 
            if a==0: #buys nothing
                # return 1 - greed(t,self.c)
                return self.R_max*(1-greed(t,self.c))
            else: #buys something
                return greed(t,self.c)*self.R_max*Phi(lambda t: sawtooth(t,self.n_actions), t, a, self.sigma) #just need to check this is right with actions-1
        else: #hungry
            if a == 0: 
                return 0 #Could also return some small number 
            else: 
                return self.R_max*Phi(lambda t: sawtooth(t,self.n_actions), t, a, self.sigma)





