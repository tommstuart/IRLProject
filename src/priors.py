import math 
import numpy as np 

#This essentially says that our only prior information is that the reward is somewhere between 0 and R_max and will ignore
#reward matrices that don't lie in this range. 
def uniform_prior_probability(R, R_max):
    return np.count_nonzero((0 < R) & (R <= R_max))/(R.size*R_max)
    # if np.count_nonzero((0<R) & (R<=R_max)) == R.size:
    #     return 1 #This should technically return 0, because the probability of choosing a random point in a uniform distribution 
    #     #across [0,R_max] is 0 because it's a continuous distribution, so I may as well just return 1. 
    # else:
    #     return 0