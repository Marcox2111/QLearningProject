import pickle
import numpy as np
from DroneEnvironment import DroneEnvironment

GRID_SIZE = 10
MAX_STEPS = 100

if __name__ == "__main__":
    # Load Value Iteration policy
    with open('value_iteration_policy.pkl', 'rb') as f:
        V, policy = pickle.load(f)
    
    # Create environment with visualization
    env = DroneEnvironment(grid_size=GRID_SIZE, max_steps=MAX_STEPS, render_mode="human", safe_return_threshold=0.3)
    state = env.reset()
    total_reward = 0.0
    
    while True:
        # Get current state components
        (r, c), steps_remaining = state
        
        # Get action from policy
        action = policy[r, c, steps_remaining]
        
        # Take step in environment
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        if done:
            print("Episode finished with reward:", total_reward)
            break
            
        state = next_state
    
    env.close()