import numpy as np 
import math
from policy import Boltzmann 


#Could just pass in the times instead of the observations. idk what gamma was, delta could be like a class variable here instead of a parameter
def policy_iteration(env, observations, delta=1e-4, pi=None):
    n_observations = len(observations) 
    #Lets set up observations to be [(s,a,t), (s,a,t),...] 
    #or well rather [ [s,a,t], [s,a,t], ..., [s,a,t]] 
    times = observations[:, 2] #extract the times of each observations

    #initialise pi randomly 
    if pi is None: 
        pi = np.random.rand(env.n_states, n_observations, env.n_actions)
        sums = np.sum(pi,axis=2)
        pi = pi/sums


    n_iter = 0
    diff = 1 
    #idk if this will work I might need to do some array shuffling after this operation 
    R = np.vectorize(env.reward)(env.states, env.actions, times) #I want this to give R = [R_t_1,...,R_t_m] where R_t_k has dimensions |S| x |A|
    values = np.random.rand(env.n_states, n_observations)#|S| x |T| so |S| x n_observations probably doesn't need to be random 

    #Policy Evaluation
    while(diff>delta): #surely this just guarantees convergence on one state-time pair 
        for s in env.n_states:
            for t in range(n_observations):
                v = values[s,t] 
                values[s,t] = compute_v_pi(env,pi,s,t,values,R)
                diff = max(diff, math.abs(v-values[s,t]))
    
    #Policy Improvement
    policy_stable = True 
    max_Q = 0 
    for s in env.n_states:
        for t in n_observations:
            b = pi[s,t] 
            q_vals = []
            for a in env.n_actions: 
                q_vals[a] = compute_q_pi(env,s,a,t,values,R)
               
            boltzmann = Boltzmann(q_vals, env.actions)
            pi[s,t] = boltzmann.getDistribution(s,t)

    return pi 
#I can do it for each element like this and then vectorize it later but it's really inefficient, I should figure out how to do it with np.sum and stuff 
def compute_v_pi(env,pi,s,t,values,R): 
    sum = 0 
    for a in env.n_actions: 
        sum += pi[s,a,t]*R[s,a,t] #this reward is from the policy walk 
    for a in env.n_actions: 
        for s_ in env.n_states: 
            sum += env.discount_rate*pi[s,a,t]*env.P[s,a,s_]*values[s_,t]

#okay you don't actually calculate the Q value of the policy 
# def compute_q_pi(env,s,a,t,values,R):
#     sum = R[s,a,t]
#     for s_ in env.n_states: 
#         sum += env.discount_rate*env.P[s,a,s_]*values[s_,t]

#this is equivalent to above if you say \forall s', s'' in S, R(s,a,s') = R(s,a,s'') or if you have determinism or something 
def compute_q_pi(env,s,a,t,values,R): 
    sum = 0
    for s_ in env.n_states: 
        sum += env.P[s,a,s_](R[s,a,s_] + env.discount_rate*values[s_,t])
