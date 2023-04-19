import numpy as np 
import learn
from policy import Boltzmann
import random 
from policy import choose_a_from_pi
#ondrej said it's the prior*likelihood. So P(R) * P(O|R) 

def policy_walk(env, observations, optimal_q_values, step_size = 0.05): #no idea what a normal step size is - they do 0.05 so I guess this is reasonable 
    n_observations = len(observations) 
    #Pick a random reward vector - I need to figure out the grid thingy 
    R = np.random.rand(env.n_states, env.n_actions, n_observations) #S x A x T
    #Perform policy iteration 
    (pi,values) = learn.policy_iteration(env, n_observations, R)

    iters = 0 
    #not sure when to stop yet? 
    while iters < 1000: 
        if (iters%100 == 0):
            print(pi)

        R_tild = get_neighbouring_reward(R, step_size) 
        (pi_tild, values_tild) = learn.policy_iteration(env, n_observations, R_tild, pi = pi)

        q_tild = np.ones((env.n_states, env.n_actions, n_observations))
        for s in range(env.n_states):
            for a in range(env.n_actions):
                for t in range(n_observations):               
                    q_tild[s,a,t] = learn.compute_q_with_values(env,s,a,t,values_tild,R_tild)

        if is_better(env, n_observations, q_tild, pi):
            ratio = calculate_posterior(env,observations, R_tild, env.R_max, pi_tild)/calculate_posterior(env,observations, R, env.R_max, pi)
            p = min(1,ratio)
            if random.random() < p:
                R = R_tild 
                pi = pi_tild 
        else:
            ratio = calculate_posterior(env,observations, R_tild, env.R_max, pi)/calculate_posterior(env,observations, R, env.R_max, pi) 
            p = min(1,ratio) 
            if random.random() < p: 
                R = R_tild 
        iters+=1 
    return R


#again their algorithm isn't really designed to work with the mapping to a distribution 
def is_better(env, n_observations, q, pi):
    for s in range(env.n_states):
        for a in range(env.n_actions): 
            for t in range(n_observations): 
                #Here I just sample an action from pi, don't know if this is what you're meant to do bc I'm using a stochastic policy 
                if q[s,pi[s,t],t] < q[s,a,t]: 
                    return True
    return False
        
#not sure whether you're meant to do it like this or shift it by the actual step size randomly in each direction 
#yeah you're not meant to do it like that - it's a mcmc step and it's meant to be more complicated I think, but I don't see why this 
#wouldn't work? 
def get_neighbouring_reward(R, step_size): 
    return R + np.random.uniform(-step_size, step_size, R.shape)

#These need changing, apparently I'm meant to use boltzmann somewhere 
#yeah not sure how to do this, surely I need the policy? But it's given the reward so maybe 
def calculate_likelihood(env, observations, R): 
    (optimal_pi, optimal_values) = learn.policy_iteration(env, len(observations), R) 
    optimal_q = np.ones((env.n_states, env.n_actions, len(observations))) 
    for s in range(env.n_states): 
        for a in range(env.n_actions): 
            for t in range(len(observations)): 
                optimal_q[s,a,t] = learn.compute_q_with_values(env,s,a,t,optimal_values,R)
    
    boltzmann = Boltzmann(optimal_q, env.actions)
    dist = boltzmann.getDistribution(optimal_q)
    product = 1 
    for (s,a,t) in observations: 
        product*=dist[s,a,t]
    return product 

#P_prior(R) * P(O|R) - not technically the posterior since I don't divide it by the probability of the observation but it doesn't matter. 
def calculate_posterior(env, observations, R, R_max, pi): 
    from priors import uniform_prior_probability 
    return uniform_prior_probability(R, R_max)*calculate_likelihood(env, observations, R)   