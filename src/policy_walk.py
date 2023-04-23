import numpy as np 
import learn
from policy import Boltzmann
import random 

def policy_walk(env, observations, n_observations, step_size = 0.05, n_iters = 10000): #no idea what a normal step size is - they do 0.05 so I guess this is reasonable 
    #Pick a random reward vector - I need to figure out the grid thingy 
    R = np.random.rand(env.n_states, env.n_actions, n_observations) #S x A x T
    #Perform policy iteration 
    acceptance_probs = np.zeros(n_iters)
    (pi,values, q_values) = learn.policy_iteration(env, n_observations, R)

    iter = 0 
    sampled_rewards = np.empty((n_iters, env.n_states, env.n_actions, n_observations))
    while iter < n_iters: 
        R_tild = get_neighbouring_reward(R, step_size) 
         #There's no harm in passing in pi, values, q_values here because otherwise they'd just end up being random 
        (pi_tild, values_tild, q_values_tild) = learn.policy_iteration(env, n_observations, R_tild, pi = pi, values = values, q_values = q_values)

        ratio = calculate_posterior(env,observations,q_values_tild, R_tild, env.R_max)/calculate_posterior(env,observations,q_values, R, env.R_max)
        p = min(1,ratio)
        acceptance_probs[iter] = p 
        if (random.random() < p):
            R = R_tild 
            values = values_tild 
            q_values = q_values_tild
            pi = pi_tild 
        sampled_rewards[iter] = R
        iter+=1 
    return sampled_rewards#, acceptance_probs

#Should be is_worse I guess - not used any more but I'll keep it in for now 
def is_better(env, n_observations, q, pi):
    for s in range(env.n_states):
        for a in range(env.n_actions): 
            for t in range(n_observations): 
                if q[s,pi[s,t],t] < q[s,a,t]: 
                    return True
    return False

def get_neighbouring_reward(R, step_size): 
    return R + np.random.uniform(-step_size, step_size, R.shape)

#is it right that we're testing likelihood wrt the boltzmann like this or should we be calculating the likelihood
#of the observation with respect to pi/pi_tild ?? 
def calculate_likelihood(env, observations,optimal_q_values): #look at doing it with log likelihoods 
    boltzmann = Boltzmann(optimal_q_values, env.actions, alpha = 5) #what alpha should we use here - if we use too low this always returns low because everything is so random/noisy
    dist = boltzmann.getDistribution(optimal_q_values)
    product = 1 
    for (s,a,t) in observations: 
        product*=dist[s,a,t]
    return product 

#P_prior(R) * P(O|R) - not technically the posterior since I don't divide it by the probability of the observation but it doesn't matter. 
def calculate_posterior(env, observations,q_values, R, R_max): 
    from priors import uniform_prior_probability 
    return uniform_prior_probability(R, R_max)*calculate_likelihood(env, observations,q_values)   