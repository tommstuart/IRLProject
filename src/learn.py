import numpy as np 
from policy import Boltzmann 

#Could just pass in the times instead of the observations, delta could be like a class variable here instead of a parameter
def policy_iteration(env, n_observations, R, delta=1e-4, pi=None, values = None, q_values = 0):

    #initialise pi randomly
    if pi is None: 
        pi = np.random.choice(env.actions, (env.n_states, n_observations)) 
    if values is None: 
        values = np.random.rand(env.n_states, n_observations)
    if q_values is None:
        q_values = np.zeros(env.n_states,env.n_actions,n_observations)
    
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
                pi[s,t] = np.argmax(q_values) 
                if b != pi[s,t]:
                    policy_stable = False
        if policy_stable == True: 
            return (pi,values,q_values)

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