import random 
from policy import choose_a_from_pi
current_time = 0.0 
def generate_trajectory(env, pi, observation_times):
    actions = []
    observations = [] 
    current_state = env.current_state 
    for t in range(len(observation_times)):
        old_state = current_state
        a = choose_a_from_pi(pi,current_state, t) #need to be careful here between using the indexes and actual times 
        (current_state, _) = env.step(current_state, a, observation_times[t]) #probably don't need these
        observations.append([old_state, a, t])
    return observations
#ugh to generate a proper policy I'll need to define my reward function, then find the Q-values of it all and then pass it into my 
#Boltzmann policy 