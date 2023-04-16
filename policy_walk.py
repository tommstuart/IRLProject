import numpy as np 
import math 
#Initialise R randomly 
#Let pi be the result of the policy iteration
#Repeat until ? : 

#Pick a neighbouring reward matrix 
# Compute Q^pi ?? Do we not do this in policy iteration ? 
# If it beats it in Q^\pi score change with probability 

#idk when we repeat that until though 
#I also need to figure out how to work out the probabilities and priors n stuff which will be tricky 
#Not sure what P(R,\pi) means tbh.  

#ondrej said it's the prior*likelihood. So P(R) * P(O|R) 
#I don't even know what prior I'm using though, is it the gaussian one? 
#I think I can just use a multivariate guassian rather than a gaussian process because 
#we've discretized the time. 
# I need to figure out where I'm going to store the priors, like do I put that in the env? Or maybe somewhere else and pass it
#into the policy walk algorithm. Like just have a classes of priors and pass it into policy walk. 

#Also, if we know the prior, should we sample the reward from the prior initially. 

def likelihood(observations, R): 
    