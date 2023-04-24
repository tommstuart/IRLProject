import math 
import numpy as np 

#This essentially says that our only prior information is that the reward is somewhere between 0 and R_max and will ignore
#reward matrices that don't lie in this range. 
def uniform_prior_probability(R, R_max):
    if np.count_nonzero((0<R) & (R<=R_max)) == R.size: # It would be neater to use np.all() here.
         return 1#/(R.size*R_max) #This should technically return 0, because the probability of choosing a random point in a uniform distribution 
        #across [0,R_max] is 0 because it's a continuous distribution, so I may as well just return 1.
        # NO! It's a density which is non-zero even if the probability of each points is technically zero.
        # Since in MCMC we're using a ratio of densities, it's fine to return 1 rather than the actual density, but you
        # risk shooting yourself in the leg if you ever want to use this function for something else.
    else:
        return 0