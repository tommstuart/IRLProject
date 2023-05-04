import numpy as np 
import learn
from policy import Boltzmann
import random 
import math

class PolicyWalk(): 
    def __init__(self, env, prior, observations, observation_times, alpha): 
        self.env = env 
        self.prior = prior 
        self.observations = observations 
        self.observation_times = observation_times 
        self.alpha = alpha 
        # self.cov = np.load('cov.npy')
        # self.rng = np.random.default_rng() 

    def get_samples(self, step_size, n_iters):
        n_observations = len(self.observation_times)
        #Initialise reward randomly 
        R = np.random.rand(self.env.n_states, self.env.n_actions, n_observations)
        # R = self.prior.sample(self.env.n_states, self.env.n_actions, n_observations)
        #Perform policy iteration         
        (pi,values, q_values) = learn.policy_iteration(self.env, n_observations, R)

        iter = 0 
        sampled_rewards = np.empty((n_iters, self.env.n_states, self.env.n_actions, n_observations))
        acceptance_probs = np.zeros(n_iters)
        while iter < n_iters: 
            #Perform mcmc step 
            R_tild = gaussian_mcmc_step(R, step_size)
            # R_tild = self.gaussian_mcmc_step(R)
            
            #There's no harm in passing in pi, values, q_values here because otherwise they'd just end up being random 
            (pi_tild, values_tild, q_values_tild) = learn.policy_iteration(self.env, n_observations, R_tild, pi = pi, values = values, q_values = q_values)
            
            tild_log_posterior = self.calculate_posterior(q_values_tild,R_tild) 
            log_posterior = self.calculate_posterior(q_values, R)

            #This is when the uniform prior part returns 0 - I think this can be omitted if we adapt mcmc step to only propose vectors within [0,R_max]
            if tild_log_posterior == -np.inf: 
                log_posterior_ratio = 0 
            else: 
                log_posterior_ratio = math.exp(tild_log_posterior - log_posterior) 
            p = min(1,log_posterior_ratio)

            acceptance_probs[iter] = p 

            if (random.random() < p):
                R = R_tild 
                values = values_tild 
                q_values = q_values_tild
                pi = pi_tild 

            sampled_rewards[iter] = R
            iter+=1 

            if math.isclose(iter/n_iters % 0.01, 0, abs_tol = 1e-8):
                print(iter/n_iters*100, end = "% ")

        return sampled_rewards, acceptance_probs

    #P_prior(R) * P(O|R) - not technically the posterior since I don't divide it by the probability of the observation but it doesn't matter. 
    def calculate_posterior(self, q_values, R):
        log_likelihood = self.calculate_log_likelihood(q_values)
        log_posterior = self.prior(R) + log_likelihood
        return log_posterior 

    def calculate_log_likelihood(self, q_values): 
        boltzmann = Boltzmann(q_values, self.env.actions, alpha = self.alpha) 
        log_likelihood = 0
        for (s,a,t) in self.observations: 
            log_likelihood += boltzmann.logdist[s,a,t]
        return log_likelihood 

    # def gaussian_mcmc_step(self,R):
    #     # return np.random.multivariate_normal(R.flatten(), self.cov).reshape(R.shape)
    #     d = np.prod(R.shape) 
    #     c = 2.4/math.sqrt(d) 
    #     #Cholesky is faster than SVD but apparently it's less robust?? I think it needs a positive definite matrix, but I'm happy to just 
    #     #find the nearest positive definite matrix before hand it doesn't make a difference for us and if it improves performance that's great. 
    #     proposed_R = self.rng.multivariate_normal(R.flatten(), np.multiply(c ** 2, self.cov), method='cholesky').reshape(R.shape)

    #     #I don't think that this is a good idea because then calculating the jumping probabilities is hard and the 
    #     #probability of jumping to a boundary matrix is way higher and would mean that the acceptance rates wouldn't change much anyway 
    #     #So I think that it would just add extra time due to the more intensive computation. 
    #     # clipped_R = np.clip(proposed_R, 0, self.env.R_max)
    #     return proposed_R

def uniform_mcmc_step(R, step_size): 
    return R + np.random.uniform(-step_size, step_size, R.shape)

#flatten it all - you want smaller perturbations in some places than others
#We should bound this by the R_max, and suggest things within the time_dependent_prior potentially because otherwise we're just wasting calculations.
def gaussian_mcmc_step(R, var):
    return np.random.normal(R,var) 