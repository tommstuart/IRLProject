import random 
current_time = 0.0 
def generate_trajectory(env, policy, n):
    actions = []
    for i in range(n): 
        current_time = current_time + random.uniform(0.0, 2.0)    
        a = policy(current_state, current_time)
        actions.append(policy(current_time))
    return actions
#ugh to generate a proper policy I'll need to define my reward function, then find the Q-values of it all and then pass it into my 
#Boltzmann policy 