import math
def Phi(theta, t, x): 
    return math.exp(-(x-theta(t)) ** 2)

def sawtooth(t, k):
    return 2*k*math.abs(t/(2*k) - math.floor(t/(2*k) + 0.5))

class SingleStateReward:
    def __init__(self, Rmax, k):
        self.Rmax = Rmax 
        self.k = k 
    def __call__(self, a, t): 
        return Phi(lambda t: sawtooth(t,self.k), t, a)
