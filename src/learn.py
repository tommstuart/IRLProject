import numpy as np 
import math
from policy import Boltzmann 
from utils import normalise_pi

#Could just pass in the times instead of the observations. idk what gamma was, delta could be like a class variable here instead of a parameter
def policy_iteration(env, observations, R, delta=1e-4, pi=None):
    n_observations = len(observations) 
    #Lets set up observations to be [(s,a,t), (s,a,t),...] - might need to be tuples to extract in a for loop - we'll see 
    #or well rather [ [s,a,t], [s,a,t], ..., [s,a,t]] 
    times = np.asarray(observations)[:, 2] #extract the times of each observations

    #initialise pi randomly - maps s,a,t to a probability value 
    if pi is None: 
        pi = np.random.rand(env.n_states, env.n_actions, n_observations)
        pi = normalise_pi(pi)
    values = np.random.rand(env.n_states, n_observations)

    iters = 0 

    while True: 
        diff = 1

        #Policy Evaluation
        while(diff > delta): #surely this just guarantees convergence on one state-time pair - no because it's the max of all of them 
            diff = 0 #the diff shrank suspiciously, like it divided by 10 each time which was weird. 10 = n_actions or something 
            for s in range(env.n_states):
                for t in range(n_observations):
                    # print(values[s,t])
                    v = values[s,t] 
                    values[s,t] = compute_v_pi(env, pi, s, t, values, R)
                    diff = max(diff, abs(v - values[s,t]))
            iters += 1 
            if iters > 1000: 
                print("failed to converge") 
                print(diff) 
        
        #Policy Improvement
        policy_stable = True 
        max_Q = 0 
        for s in range(env.n_states):
            for t in range(n_observations):
                b = pi[s,:,t]
                q_vals = np.zeros(env.n_actions)
                for a in range(env.n_actions): 
                    q_vals[a] = compute_q_with_values(env,s,a,t,values,R)

                #normally they take the max of the q_values and map the policy to that action.
                #But since we're trying to maintain pi as a distribution, how do we convert the Q-values 
                #back into a distribution? Do I just normalise or do I arrange in a boltzmann distribution? 
                boltzmann = Boltzmann(q_vals, env.actions) #idk if I'm doing this right even if this is what I'm meant to be doing 
                pi[s,:,t] = boltzmann.getDistribution(q_vals) #ye this just doesn't work idk what I'm trying to do 
                if np.linalg.norm(b - pi[s,:,t], ord = 1) >= 0.01: #in their code they wait until pi stops changing it's mind about specific actions but we can't do that because we use stochastic policies. This is crude but it'll do for now. 
                    policy_stable = False
        if policy_stable == True: 
            return (pi,values)
        else:
            # Go back to Policy Evaluation
            iters = 0 
            diff = 1
            continue


#I can do it for each element like this and then vectorize it later but it's really inefficient, I should figure out how to do it with np.sum and stuff 
def compute_v_pi(env,pi,s,t,values,R): 
    sum = 0 
    for a in range(env.n_actions): 
        sum += pi[s,a,t]*R[s,a,t] #this reward is from the policy walk 
    for a in range(env.n_actions): 
        for s_ in range(env.n_states): 
            sum += env.discount_rate*pi[s,a,t]*env.P[s,a,s_]*values[s_,t]
    return sum 

#okay you don't actually calculate the Q value of the policy 
# def compute_q_pi(env,s,a,t,values,R):
#     sum = R[s,a,t]
#     for s_ in env.n_states: 
#         sum += env.discount_rate*env.P[s,a,s_]*values[s_,t]

#this is equivalent to above if you say \forall s', s'' in S, R(s,a,s') = R(s,a,s'') or if you have determinism or something 
def compute_q_with_values(env,s,a,t,values,R): 
    sum = 0
    for s_ in range(env.n_states): 
        sum += env.P[s,a,s_]*(R[s,a,s_] + env.discount_rate*values[s_,t])
    return sum 
def compute_q_with_pi(env,s,a,t,pi,values,R):
    # values = compute_v_pi(env, pi, s, t, values, R) #no idea if this works otherwise re-run the stuff in policy eval etc., surely that can't work.
    return compute_q_with_values(env, s,a,t,values,R)

