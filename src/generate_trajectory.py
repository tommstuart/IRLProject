current_time = 0.0 
def generate_trajectory(env, pi, observation_times):
    actions = []
    observations = [] 
    current_state = env.current_state 
    for t in range(len(observation_times)):
        old_state = current_state
        a = pi(current_state,t)
        (current_state, _) = env.step(current_state, a, observation_times[t])
        observations.append([old_state, a, t])
    return observations