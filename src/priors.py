import math 
import numpy as np 
#So previously, our prior knowledge was that it's a drifting gaussian, but for this don't even need to know that 
#we could just use a completely general prior? Yeah we need some prior information to know how to calculate the likelihood 
#of the reward? 
#R is a matrix over |S|x|A|x|T| but in reality it's going to be |S| x |A| x n_observations 
# def gaussian_prior(R, mu, sigma): 
#     #it'll be like the average over S, A and T of the values ugh idk how did I define this in the note s
#     return 3

def uniform_prior_probability(R, R_max):
    #surely this is just going to always return a really low number? 
    return np.count_nonzero(0 <= R <= R_max)/(R.size*R_max)