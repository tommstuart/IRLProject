from RewardFunctions import SingleStateReward
import gymnasium as gym 
import numpy as np 

# class StateSpace(gym.Env): - I could define some parent class which implements the generalised step etc. because they'll all be the same 

class SingleStateSpace(gym.Env): 
    def __init__(self, k, discount_rate, R_max): 
        self.reward = SingleStateReward(self, Rmax = 10, k = k) 
        self.k = k
        self.R_max = R_max 
        self.discount_rate = discount_rate 

        self.states = [0] 
        self.n_states = 1
        self.actions = list(range(k))
        self.n_actions = k 

        self.P = np.ones(1,self.n_actions,1) 

    def step(self, s, a ,t): 
        assert 0<=a<self.k 
        reward = self.reward(s,a,t) 
        next_state = np.random.choice([*range(self.n_states)], self.P[s,a,:]) #randomly choose a next state according to the transition model 
        return (next_state, reward) 
    
    def reset(self): 
        self.states = [0] 


    #Doesn't need to do anything here 
    #do I need render/close 