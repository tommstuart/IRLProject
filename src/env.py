from RewardFunctions import SingleStateReward
import gymnasium as gym 

class SingleStateSpace(gym.Env): 
    def __init__(self, k, Rmax): 
        self.reward = SingleStateReward(self, Rmax = 10, k = k) 
        self.k = k
        self.Rmax = Rmax 
        self.states = [0] 
        self.actions = [0..k-1] # I mean this probably doesn't work but you get the point - I need to declare this here I think 

    def step(self, s, a ,t): 
        assert 0<=a<self.k 
        reward = self.reward(a,t) 
        return (0, reward) 
    
    def reset(self): 
        self.states = [0] 


    #Doesn't need to do anything here 
    #do I need render/close 