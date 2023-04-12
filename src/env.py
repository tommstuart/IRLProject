from RewardFunctions import SingleStateReward

class SingleStateSpace():
    #k is the number of actions
    def __init__(self, k, Rmax):
        self.reward = SingleStateReward(self, Rmax = 10, k = k)
        self.k = Rmax 
        self.states = [0]
    
    def do_action(self, s, a, t): 
        assert 0<=a<self.k
        reward = self.reward(a, t)

        return reward #don't need to update state space because there's only one. 