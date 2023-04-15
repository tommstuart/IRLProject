import numpy as np 
import math


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

    #Randomly initialise V^\pi as well?? 

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
            for a in env.n_actions: 
                #also they don't do the Q-value, they do it without R(s,a,t), they just do the sum bit? This max business could be done with a vectorize and an argmax 
                q_val = compute_q_pi(env,s,a,t,values,R)
                if q_val > max_Q:
                    max_a = a 
                    max_Q = q_val 
            pi[s,t] = max_a #oh bc their policy maps a state to an action but I need to map it to a distribution.
            #but then I'd need to know that they're doing boltzmann? 
            #I could just set the distribution to the boltzmann of the Q-values? 

            compute_q_pi()

#I can do it for each element like this and then vectorize it later but it's really inefficient, I should figure out how to do it with np.sum and stuff 
def compute_v_pi(env,pi,s,t,values,R): 
    sum = 0 
    for a in env.n_actions: 
        sum += pi[s,a,t]*R[s,a,t] #not sure if this is env.reward, or probs should be the reward from the policy walk 
    for a in env.n_actions: 
        for s_ in env.n_states: 
            sum += env.discount_rate*pi[s,a,t]*env.P[s,a,s_]*values[s_,t]

def compute_q_pi(env,s,a,t,values,R):
    sum = R[s,a,t]
    for s_ in env.n_states: 
        sum += env.discount_rate*env.P[s,a,s_]*values[s_,t]
