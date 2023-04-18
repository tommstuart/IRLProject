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
    (pi,values) = learn.policy_iteration(env, observations, R)

    iters = 0 
    #not sure when to stop yet? 
    while iters < 1000: 
        if (iters%100 == 0):
            print(pi)
        R_tild = get_neighbouring_reward(R, step_size) 
        q = np.ones((env.n_states, env.n_actions, n_observations))
        # values = np.ones((env.n_states, env.n_actions, n_observations))
        #should rlly vectorize more stuff but this is fine for now it makes it clearer
        for s in range(env.n_states):
            for a in range(env.n_actions):
                for t in range(n_observations):               
                    q[s,a,t] = learn.compute_q_with_pi(env,s,a,t,pi,values,R_tild)

        if is_better(env, n_observations, q, pi):
            (pi_tild, values_tild) = learn.policy_iteration(env, observations, R_tild, pi = pi)
            ratio = calculate_posterior(observations, R_tild, env.R_max, pi_tild)/calculate_posterior(observations, R, env.R_max, pi)
            p = min(1,ratio)
            if random.random() < p:
                R = R_tild 
                pi = pi_tild 
        else:
            ratio = calculate_posterior(observations, R_tild, env.R_max, pi)/calculate_posterior(observations, R, env.R_max, pi) 
            p = min(1,ratio) 
            if random.random() < p: 
                R = R_tild 
        iters+=1 
    return pi


#again their algorithm isn't really designed to work with the mapping to a distribution 
def is_better(env, n_observations, q, pi):
    for s in range(env.n_states):
        for a in range(env.n_actions): 
            for t in range(n_observations): 
                #Here I just sample an action from pi, don't know if this is what you're meant to do bc I'm using a stochastic policy 
                if q[s,choose_a_from_pi(pi,s,t),t] < q[s,a,t]: 
                    return True
    return False
        
#not sure whether you're meant to do it like this or shift it by the actual step size randomly in each direction 
def get_neighbouring_reward(R, step_size): 
    return R + np.random.uniform(-step_size, step_size, R.shape)

def calculate_likelihood(observations, pi): 
    product = 1
    for [s,a,t] in observations: #might need to do tuples 
        product *= pi[s,a,t]
    return product

#P_prior(R) * P(O|R) - not technically the posterior since I don't divide it by the probability of the observation but it doesn't matter. 
def calculate_posterior(observations, R, R_max, pi): 
    from priors import uniform_prior_probability 
    return uniform_prior_probability(R, R_max)*calculate_likelihood(observations,pi)   