from env import SingleStateSpace 
from generate_trajectory import generate_trajectory
from policy import Boltzmann 
from learn import compute_q_with_pi
from RewardFunctions import SingleStateReward
import numpy as np 
from utils import normalise_pi
from policy_walk import policy_walk 
# def run_tests(trajectory_length = 20): 

trajectory_length = 20 
#set up the agent and their policy 
env = SingleStateSpace(n_actions = 10, discount_rate = 0.1, R_max = 5)
#so we have access to env.reward
observation_times = np.cumsum(np.random.uniform(0, 2, size=trajectory_length))
print("Generated observation times:") 
print(observation_times)

#this isn't right but for now I'm just normalising pi - really it should be a boltzmann - later on I turn the Q-vals into a distribution via boltzmann 
#so it won't be able to learn i don't think 


# Use meshgrid to create a grid of all possible combinations of s, a, and t
s_grid, a_grid, t_grid = np.meshgrid(env.states, env.actions, observation_times, indexing='ij')

# Calculate the reward for each combination using vectorized operations
pi = np.vectorize(env.reward)(s_grid, a_grid, t_grid)
pi = normalise_pi(pi)

#What is my initial policy? Everywhere I've seen, boltzmann is defined using the Q-values but you
#need a policy to find the Q-values so what is the actual policy? 
#Is it just like an exponential style distribution over the rewards? 
observations = generate_trajectory(env, pi, observation_times)

print("Generated trajectory") 
print(observations)

print("Running policy walk") 
learned_pi = policy_walk(env, observations)
print("finished learning policy")

#generate the trajectory 
#run birl on the trajectory 
#take the reward and the initial reward and compare 


#I'm gonna need prints/logs to see how badly it converges. 
