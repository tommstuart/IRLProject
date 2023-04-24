import numpy as np 
import learn
from policy import Boltzmann
import random 

# re step size: see the tips from my email from last week
def policy_walk(env, observations, n_observations, step_size = 0.05, n_iters = 10000): #no idea what a normal step size is - they do 0.05 so I guess this is reasonable 
    #Pick a random reward vector - I need to figure out the grid thingy 
    R = np.random.rand(env.n_states, env.n_actions, n_observations) #S x A x T
    #Perform policy iteration 
    (pi, values, q_values) = learn.policy_iteration(env, n_observations, R)

    iters = 0 
    sampled_rewards = [] 
    while iters < n_iters: 
        R_tild = get_neighbouring_reward(R, step_size) 
        pi_tild, values_tild, q_values_tild = learn.policy_iteration(env, n_observations, R_tild, pi = pi, values = values, q_values = q_values)

        #Maybe do value iteration i.e. combine the two loops of policy iteration and then you don't need to do this check because the policy you compute will be optimal
        # Yes. Please pass the q_values to calculate_posterior and use that to compute the likelihood.
        ratio = calculate_posterior(env,observations,n_observations, R_tild, env.R_max)/calculate_posterior(env,observations, n_observations, R, env.R_max)
        p = min(1,ratio)
        if (random.random() < p):
            R = R_tild 
            values = values_tild 
            q_values = q_values_tild
            pi = pi_tild 
        iters+=1 
        sampled_rewards.append(R)
    return sampled_rewards

#Should be is_worse I guess
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
def calculate_likelihood(env, observations, n_observations, R): #look at doing it with log likelihoods 
    (optimal_pi, optimal_values, optimal_q_values) = learn.policy_iteration(env, n_observations, R) 
    
    boltzmann = Boltzmann(optimal_q_values, env.actions, alpha = 5) #what alpha should we use here - if we use too low this always returns low because everything is so random/noisy  A: 5 sounds ok. Make sure you always use the same one here and to generate the demonstrations.
    dist = boltzmann.getDistribution(optimal_q_values)
    product = 1 
    for (s,a,t) in observations: 
        product*=dist[s,a,t]
    return product 

#P_prior(R) * P(O|R) - not technically the posterior since I don't divide it by the probability of the observation but it doesn't matter. 
def calculate_posterior(env, observations, n_observations, R, R_max): 
    from priors import uniform_prior_probability  # Move this import to the top of the file
    return uniform_prior_probability(R, R_max)*calculate_likelihood(env, observations, n_observations, R)   