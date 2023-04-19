import numpy as np 
import math
from policy import Boltzmann 
from utils import normalise_pi

#Could just pass in the times instead of the observations. idk what gamma was, delta could be like a class variable here instead of a parameter
def policy_iteration(env, n_observations, R, delta=1e-4, pi=None):

    #initialise pi randomly
    if pi is None: 
        pi = np.random.choice(env.actions, (env.n_states, n_observations)) 
    values = np.random.rand(env.n_states, n_observations)
    
    while True: 
        diff = 1
        iters = 0
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
        print(values)
        #Policy Improvement
        policy_stable = True 
        max_Q = 0 
        for s in range(env.n_states):
            for t in range(n_observations):
                b = pi[s,t]   
                q_vals = np.zeros(env.n_actions)
                for a in range(env.n_actions): 
                    q_vals[a] = compute_q_with_values(env,s,a,t,values,R)
                pi[s,t] = np.argmax(q_vals) 
                if b != pi[s,t]:
                    policy_stable = False
        if policy_stable == True: 
            return (pi,values)

def compute_v_pi(env,pi,s,t,values,R):
    sum = 0 
    for s_ in range(env.n_states): 
        sum += env.P[s,pi[s,t],s_]*(R[s,pi[s,t],s_] + env.discount_rate*values[s_,t])
    return sum

#this is equivalent to above if you say \forall s', s'' in S, R(s,a,s') = R(s,a,s'') or if you have determinism or something 
def compute_q_with_values(env,s,a,t,values,R): 
    sum = 0
    for s_ in range(env.n_states): 
        sum += env.P[s,a,s_]*(R[s,a,t] + env.discount_rate*values[s_,t])
    return sum 

#I need to get a function which just returns the whole Q matrix rather than keep doing a triple loop and calling this it's daft 
    
    
# def compute_q_with_pi(env,s,a,t,pi,values,R):
#     # values = compute_v_pi(env, pi, s, t, values, R) #no idea if this works otherwise re-run the stuff in policy eval etc., surely that can't work.
#     return compute_q_with_values(env, s,a,t,values,R)

