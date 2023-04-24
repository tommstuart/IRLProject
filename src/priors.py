import math 
import numpy as np 

#This essentially says that our only prior information is that the reward is somewhere between 0 and R_max and will ignore
#reward matrices that don't lie in this range. 
def uniform_prior_probability(R, R_max):
    if np.all((0<R) & (R<=R_max)):
        #returning 1 here only works because we're computing a ratio. 
        return 1 
    else: 
        return 0