import numpy as np 
import learn
from policy import Boltzmann
import random 
#ondrej said it's the prior*likelihood. So P(R) * P(O|R) 

def policy_walk(env, observations, step_size = 0.01): #no idea what a normal step size is  
    n_observations = len(observations) 
    #Pick a random reward vector - I need to figure out the grid thingy 
    R = np.random.rand(env.n_states, env.n_actions, n_observations) #S x A x T
    #Perform policy iteration 
    pi = learn.policy_iteration(env, observations, R)

    #not sure when to stop yet? 
    while True: 
        R_tild = get_neighbouring_reward(R, step_size) 
        q = np.ones(env.n_states, env.n_actions, n_observations)
        values = np.ones(env.n_states, env.n_actions, n_observations)
        #should rlly vectorize more stuff but this is fine for now it makes it clearer
        for s in env.n_states:
            for a in env.n_actions:
                for t in n_observations:                 
                    #idk what to do here, am I meant to be assuming pi is a boltzmann policy instance
                    q[s,a,t] = learn.compute_q_with_pi(env,s,a,t,pi,R_tild)
        boltzmann_policy = Boltzmann(q) 
        if is_better(env, n_observations, q, boltzmann_policy):
            pi_tild = learn.policy_iteration(env, observations, R_tild, pi = pi)
            ratio = calculate_posterior(R_tild, pi_tild)/calculate_posterior(R,pi)
            p = min(1,ratio)
            if random.random() < p:
                R = R_tild 
                pi = pi_tild 
        else:
            ratio = calculate_posterior(R_tild, pi)/calculate_posterior(R, pi) 
            p = min(1,ratio) 
            if random.random() < p: 
                R = R_tild 


#again their algorithm isn't really designed to work with the mapping to a distribution 
#if exists (s,a) s.t. Q(s,pi(s),R_tild) < Q(s,a,R_tild)
def is_better(env, n_observations, q,pi):
    for s in env.n_states:
        for a in env.n_actions: 
            for t in n_observations: 
                if q[s,pi(s,t),t] < q[s,a,t]: #this is dodgy and idk if it's what I'm meant to do
                    return True
    return False
        
#not sure whether you're meant to do it like this or shift it by the actual step size randomly in each direction 
def get_neighbouring_reward(R, step_size): 
    return R + np.random.uniform(-step_size, step_size, R.shape)

#P_prior(R) * P(O|R)
def calculate_posterior(R, pi): 
