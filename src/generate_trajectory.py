import random 
current_time = 0.0 
def generate_trajectory(env, policy, n):
    actions = []
    for i in range(n): 
        current_time = current_time + random.uniform(0.0, 2.0)    
        a = policy(current_time) 
        
        actions.append(policy(current_time))
    return actions
