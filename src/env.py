from RewardFunctions import SingleStateReward
from RewardFunctions import DoubleStateReward
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

class DoubleStateSpace(gym.Env): 
    def __init__(self, n_actions, discount_rate, greed_longevity = 3, satiated_prob = 0.5, hunger_prob = 0.5, R_max=5): 
        #I think I need to call __init__ on the super class 
        assert(n_actions >= 2) #We need at least a do nothing and a buy something action 
        self.reward = DoubleStateReward(R_max, n_actions, greed_longevity)
        self.R_max = R_max 
        self.discount_rate = discount_rate 
        self.satiated_prob = satiated_prob
        self.hunger_prob = hunger_prob

        self.states = [0,1] 
        self.n_states = 2
        self.actions = list(range(n_actions))
        self.n_actions = n_actions

        self.current_state = 0 #Start off not hungry 

        self.P = self.constructP()

    def constructP(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in self.states: 
            for a in self.actions: 
                for s_ in self.states:
                    #Going from not hungry -> not hungry 
                    if (s,s_) == (0,0):
                        if a == 0:
                            P[s,a,s_] = 1 - self.hunger_prob 
                        else:
                            P[s,a,s_] = 1
                    #Going from not hungry -> hungry 
                    elif (s,s_) == (0,1): 
                        if a == 0: 
                            P[s,a,s_] = self.hunger_prob 
                        else: 
                            P[s,a,s_] = 0
                    #Going from hungry -> not hungry 
                    elif (s,s_) == (1,0):
                        if a == 0: 
                            P[s,a,s_] = 0
                        else: 
                            P[s,a,s_] = self.satiated_prob
                    #Going from hungry -> hungry 
                    else:
                        if a == 0: 
                            P[s,a,s_] = 1                            
                        else: 
                            P[s,a,s_] = 1 - self.satiated_prob 
        return P


    def step(self, s, a ,t): 
        assert 0<=a<self.n_actions 
        reward = self.reward(s,a,t) 
        next_state = np.random.choice([*range(self.n_states)], p = self.P[s,a,:]) #randomly choose a next state according to the transition model 
        self.current_state = next_state 
        return (next_state, reward) 
    
    def reset(self): 
        self.current_state = 0 