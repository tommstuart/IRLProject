from env import SingleStateSpace 
from generate_trajectory import generate_trajectory
from policy import Boltzmann 
from learn import compute_q_with_pi
import numpy as np 
def run_tests(trajectory_length = 20): 
    #set up the agent and their policy 
    env = SingleStateSpace(k = 10, discount_rate = 0.1, R_max = 10)
    
    q = np.ones(env.n_states, env.n_actions, trajectory_length)

    for s in env.n_states: 
        for a in env.n_actions: 
            for 

    q = compute_q_with_pi(env, )
    policy = Boltzmann(q) 
    observations = generate_trajectory(env, policy, n=trajectory_length)
    
    #generate the trajectory 
    #run birl on the trajectory 
    #take the reward and the initial reward and compare 
    

    #I'm gonna need prints/logs to see how badly it converges. 
