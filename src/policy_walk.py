import numpy as np 
import learn
from policy import Boltzmann
import random 

class PolicyWalk(): 
    def __init__(self, env, prior, observations, observation_times, alpha): 
        self.env = env 
        self.prior = prior 
        self.observations = observations 
        self.observation_times = observation_times 
        self.alpha = alpha 

    def get_samples(self, step_size, n_iters):
        n_observations = len(self.observation_times)
        #Pick a random reward vector
        R = np.random.rand(self.env.n_states, self.env.n_actions, n_observations) #S x A x T
        #Perform policy iteration 
        acceptance_probs = np.zeros(n_iters)
        (pi,values, q_values) = learn.policy_iteration(self.env, n_observations, R)

        iter = 0 
        sampled_rewards = np.empty((n_iters, self.env.n_states, self.env.n_actions, n_observations))
        while iter < n_iters: 

            R_tild = gaussian_mcmc_step(R, step_size)
            
            #There's no harm in passing in pi, values, q_values here because otherwise they'd just end up being random 
            (pi_tild, values_tild, q_values_tild) = learn.policy_iteration(self.env, n_observations, R_tild, pi = pi, values = values, q_values = q_values)

            ratio = self.calculate_posterior(q_values_tild, R_tild)/self.calculate_posterior(q_values, R)
            p = min(1,ratio)
            acceptance_probs[iter] = p 
            if (random.random() < p):
                R = R_tild 
                values = values_tild 
                q_values = q_values_tild
                pi = pi_tild 
            sampled_rewards[iter] = R
            iter+=1 
        return sampled_rewards, acceptance_probs

    #P_prior(R) * P(O|R) - not technically the posterior since I don't divide it by the probability of the observation but it doesn't matter. 
    def calculate_posterior(self, q_values, R):
        return self.prior(R)*self.calculate_likelihood(q_values)

    def calculate_likelihood(self, q_values): 
        boltzmann = Boltzmann(q_values, self.env.actions, alpha = self.alpha) 
        product = 1 
        for (s,a,t) in self.observations: 
            product *= boltzmann.dist[s,a,t]
        return product 

def uniform_mcmc_step(R, step_size): 
    return R + np.random.uniform(-step_size, step_size, R.shape)

#flatten it all - you want smaller perturbations in some places than others
#We should bound this by the R_max, and suggest things within the time_dependent_prior potentially because otherwise we're just wasting calculations.
def gaussian_mcmc_step(R, var):
    return np.random.normal(R,var) 