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
    # When you have not yet established a reasonable default for alpha, maybe it's better no to set it at all
    # so that it forces you to choose it and it doesn't backfire as it did in your case
    def __init__(self, q, actions, alpha=0.5): #larger alpha means agent sticks to reward more
        super(Boltzmann, self).__init__(q, actions) 
        self.temp = 1/alpha 
        self.actions = actions

    # The convention is generally to write comments on the previous line, so that a reader naturally first reads the
    # comment and then the code (which the comment makes easier to understand)
    def __call__(self, s, t): #This is normalised wrong but it doesn't make a difference for now, the np.randomchoice figures it all out. 
        q_vals = []
        # I'd recommend rewriting everything in terms of numpy arrays; vectorized whenever possible (or writing
        # directly in torch if there's a chance you may need automatic differentiation, though I guess you
        # might not need it in this project)
        q_vals = self.q[s, self.actions, t]

        exp_q = np.exp(np.asarray(q_vals)/self.temp) 
        z = np.sum(exp_q) 
        return np.random.choice(self.actions, p = exp_q/z)
    
    def getDistribution(self, q_vals):
        exp_q = np.exp(q_vals/self.temp)
        sums = np.sum(exp_q, axis = 1) 
        sums = sums[:,np.newaxis, :] 
        return exp_q/sums 