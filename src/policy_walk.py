import numpy as np 
import learn
from policy import Boltzmann
import random 
from policy import choose_a_from_pi
#ondrej said it's the prior*likelihood. So P(R) * P(O|R) 

def policy_walk(env, observations, step_size = 0.05): #no idea what a normal step size is - they do 0.05 so I guess this is reasonable 
    n_observations = len(observations) 
    #Pick a random reward vector - I need to figure out the grid thingy 
    R = np.random.rand(env.n_states, env.n_actions, n_observations) #S x A x T
    #Perform policy iteration 
    (pi,values, q_values) = learn.policy_iteration(env, n_observations, R)

    iters = 0 
    sampled_rewards = [] 
    while iters < 10000: 
        R_tild = get_neighbouring_reward(R, step_size) 
        #I can just pass in the previous values array and then not have to generate it randomly at the start of each 
        #policy iteration 
        (pi_tild, values_tild, q_values_tild) = learn.policy_iteration(env, n_observations, R_tild, pi = pi, values = values, q_values = q_values)

        #Maybe do value iteration i.e. combine the two loops of policy iteration and then you don't need to do this check because the policy you compute will be optimal 
        #
        # if is_better(env, n_observations, q_tild, pi):
        ratio = calculate_posterior(env,observations, R_tild, env.R_max, pi_tild)/calculate_posterior(env,observations, R, env.R_max, pi)
        p = min(1,ratio)
        if random.random() < p:
            R = R_tild 
            values = values_tild 
            q_values = q_values_tild
            pi = pi_tild 
        # else:
        #     ratio = calculate_posterior(env,observations, R_tild, env.R_max, pi)/calculate_posterior(env,observations, R, env.R_max, pi) 
        #     p = min(1,ratio) 
        #     if random.random() < p: 
        #         R = R_tild 
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

#this is probably wrong
def get_neighbouring_reward(R, step_size): 
    return R + np.random.uniform(-step_size, step_size, R.shape)

def calculate_likelihood(env, observations, R): #look at doing it with log likelihoods 
    (optimal_pi, optimal_values, optimal_q_values) = learn.policy_iteration(env, len(observations), R) 
    
    boltzmann = Boltzmann(optimal_q_values, env.actions)
    dist = boltzmann.getDistribution(optimal_q_values)
    product = 1 
    for (s,a,t) in observations: 
        product*=dist[s,a,t]
    return product 

#Not sure if my posterior/likelihood calculations are correct
#P_prior(R) * P(O|R) - not technically the posterior since I don't divide it by the probability of the observation but it doesn't matter. 
def calculate_posterior(env, observations, R, R_max, pi): 
    from priors import uniform_prior_probability 
    return uniform_prior_probability(R, R_max)*calculate_likelihood(env, observations, R)   