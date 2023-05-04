import numpy as np 

#Essentially just says that our reward lies somewhere between 0 and R_max 
class UniformPrior(): 
    def __init__(self, R_max): 
        self.R_max = R_max 
    def __call__(self, R):
        return uniform_prior_probability(R,self.R_max)
    def sample(self, shape):
        return self.R_max*np.random.rand(shape) 

class TimeDependentPrior():
    def __init__(self, observation_times, R_max, sigma): 
        self.R_max = R_max 
        self.observation_times = observation_times 
        self.n_observations = len(observation_times)
        self.sigma = sigma 
        
    #Returns log of the prior 
    def __call__(self, R): 
        #this is fine since uniform_prior returns 0 or 1 
        if uniform_prior_probability(R,self.R_max) == 0: 
            return -np.inf
        else: 
            return self.closeness_requirement(R)
    
    #I don't know if this will introduce too much bias or not.
    #Randomly sample the first time steps vector and then iterate through, generating the next time steps vector from the previous one
    #to achieve a vector that should satisfy the prior. 
    def sample(self, n1, n2, n3):
        R = self.R_max/2 * np.random.rand(n1,n2,n3) #then I want to coerce it into the correct form 
        for i in range(self.n_observations-1): 
            time_diff = abs(self.observation_times[i] - self.observation_times[i+1])
            cov = (time_diff * self.sigma)*np.identity(R[:,:,0].size)
            
            R[:,:,i+1] = np.random.multivariate_normal(R[:,:,i].flatten(), cov).reshape(R[:,:,i].shape)
            R[:,:,i+1] = np.clip(R[:,:,i+1], 0, self.R_max) #not sure what to do here - it doesn't really matter for this 
            #but it's not really a true sample
        return R/2

    #Centred at 0 - I've optimised this for using a constant variance rather than a cov matrix 
    def unnormalised_gaussian(self,x,cov_inv): 
        x_flat = x.flatten()
        #(x_flat - 0).T * Sigma^-1 * (x_flat)
        #Assuming cov_inv is a scalar here for now 
        return np.exp(-0.5*np.dot(x_flat,x_flat)*cov_inv)
        # return np.exp(-0.5*np.matmul(np.matmul((x_flat-mean).T, cov_inv), x_flat-mean))

    def log_gaussian(self,x,cov_inv): 
        x_flat = x.flatten() 
        return -0.5*np.dot(x_flat, x_flat)*cov_inv

    def closeness_requirement(self, R): 
        sum = 0 

        #Just doing pairwise now for performance sake 
        for i in range(self.n_observations-1): 
            # for j in range(n_observations): 
            #     if i != j: 
            x = R[:,:,i] - R[:,:,i+1]
            time_diff = abs(self.observation_times[i] - self.observation_times[i+1])
            cov_inv = 1/(time_diff*self.sigma)
            sum += self.log_gaussian(x,cov_inv)
        return sum 
            

def uniform_prior_probability(R, R_max):
    if np.all((0<=R) & (R<=R_max)):
        return 1 #returning 1 here only works because we're computing a ratio. 
    else: 
        return 0