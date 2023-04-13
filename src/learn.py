import numpy as np 
import copy 

#policy is a map from S X T to a distribution over actions 
#i.e. a map from S X T to [0,k) intersect ints 
def policy_iteration(env, gamma, pi = None): 
    if pi is None: 
        gamma 
    count = 0

    #state_values = env.rewards - i.e. they make a copy 
    #of the values for each state 
    while True:
        old_pi = copy.deepcopy(pi) 
        #state values = compute v for pi for each state 
        #so we pick the pi with the best q value for the v 
        #if pi is the same as old pi return otherwise keep iterating .

#for them pi is a vector of the q values for each state. For us pi will be a tensor 
#pi is a map from (s,t) to a probability distribution over actions. 
#so we can think of it as the q values for the actions, so ours will map time to a vector of q values for each action 

def policy_iteration(env, gamma, pi=None):
    if pi is None:
        pi = np.random.randint(env.n_actions, size=env.n_states)
    n_iter = 0
    state_values = copy.deepcopy(env.rewards)
    while True:
        old_pi = copy.deepcopy(pi)
        state_values = compute_v_for_pi(env, pi, gamma, state_values)
        pi = np.argmax(__compute_q_with_v(env, state_values, gamma), axis=1)
        if np.all(old_pi == pi):
            return pi
        else:
            n_iter += 1
            if n_iter > 1000:
                print('n_iter: ', n_iter)
                print('rewards: ', env.rewards)
