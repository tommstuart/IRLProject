import numpy as np 
from policy import Boltzmann 
import copy

def policy_iteration(env, n_observations, R, delta=1e-4, pi=None, values = None, q_values = None):

    #Normally we pass in these to save recalculating them, but on the first call we randomly initialise pi and values.
    if pi is None: 
        pi = np.random.choice(env.actions, (env.n_states, n_observations)) 
    else:
        pi = copy.copy(pi)
    if values is None: 
        values = np.random.rand(env.n_states, n_observations)
    else:
        values = copy.copy(values)
    if q_values is None:
        q_values = np.zeros((env.n_states,env.n_actions,n_observations))
    else: 
        q_values = copy.copy(q_values) 
    
    
    while True: 
        diff = 1

        #Policy Evaluation
        while(diff > delta):
            diff = 0
            for s in range(env.n_states):
                for t in range(n_observations):
                    v = values[s,t] 
                    values[s,t] = compute_v_pi(env, pi, s, t, values, R)
                    diff = max(diff, abs(v - values[s,t]))

        #Policy Improvement
        policy_stable = True 
        for s in range(env.n_states):
            for t in range(n_observations):
                b = pi[s,t]   
                for a in range(env.n_actions): 
                    q_values[s,a,t] = compute_q_with_values(env,s,a,t,values,R)
                pi[s,t] = np.argmax(q_values[s,:,t]) 
                if b != pi[s,t]:
                    policy_stable = False
        if policy_stable: 
            return (pi,values,q_values)

def compute_v_pi(env,pi,s,t,values,R):
    sum = 0 
    for s_ in range(env.n_states): 
        sum += env.P[s,pi[s,t],s_]*(R[s,pi[s,t],t] + env.discount_rate*values[s_,t])
    return sum

#this is equivalent to above if you say \forall s', s'' in S, R(s,a,s') = R(s,a,s'') or if you have determinism or something 
def compute_q_with_values(env,s,a,t,values,R): 
    sum = 0
    for s_ in range(env.n_states): 
        sum += env.P[s,a,s_]*(R[s,a,t] + env.discount_rate*values[s_,t])
    return sum 