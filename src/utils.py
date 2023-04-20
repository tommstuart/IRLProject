import numpy as np 
#this isn't used at the minute 
#expects pi[s,a,t], normalised along the actions to make into a prob distribution
def normalise_pi(pi): #this could go in policy really 
    sums = np.sum(pi, axis = 1)
    sums = sums[:,np.newaxis,:] 
    return pi/sums 