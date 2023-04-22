from RewardFunctions import SingleStateReward
import gymnasium as gym 
import numpy as np 

# class StateSpace(gym.Env): - I could define some parent class which implements the generalised step etc. because they'll all be the same 

class SingleStateSpace(gym.Env): 
    def __init__(self, n_actions, discount_rate, R_max=5): 
        #I think I need to call __init__ on the super class 
        self.reward = SingleStateReward(R_max = R_max, n_actions = n_actions) 
        self.R_max = R_max 
        self.discount_rate = discount_rate 

        self.states = [0] 
        self.n_states = 1
        self.actions = list(range(n_actions))
        self.n_actions = n_actions

        self.current_state = 0 

        self.P = np.ones((1,self.n_actions,1))

    def step(self, s, a ,t): 
        assert 0<=a<self.n_actions 
        reward = self.reward(s,a,t) 
        next_state = np.random.choice([*range(self.n_states)], p = self.P[s,a,:]) #randomly choose a next state according to the transition model 
        # self.current_state = next_state # unnecessary here - single state 
        return (next_state, reward) 
    
    def reset(self): 
        self.states = [0] 
        self.current_state = 0 