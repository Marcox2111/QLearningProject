import pickle
import numpy as np
from DroneEnvironment import DroneEnvironment
import csv

GRID_SIZE = 10
MAX_STEPS = 70

if __name__ == "__main__":
    # 1) Load the GA policy (no "rewards" tuple this time)
    with open('best_ga_policy.pkl', 'rb') as f:
        ga_policy = pickle.load(f)
        # ga_policy should be shape (GRID_SIZE, GRID_SIZE, MAX_STEPS+1)
        # Each entry ga_policy[r,c,s] is an integer in [0..3]

    # 2) Create the environment for visualization
    env = DroneEnvironment(
        grid_size=GRID_SIZE, 
        max_steps=MAX_STEPS, 
        render_mode="human", 
        safe_return_threshold=0.3
    )
    state = env.reset()  # state is ((r, c), steps_remaining)
    total_reward = 0.0

    #save ga policy as a csv (a 3d array)
    while True:
        (r, c), steps_remaining = state
        # 3) Look up the action directly from the GA policy
        steps_used = MAX_STEPS - steps_remaining

        action = ga_policy[r, c, steps_used]
        # action = ga_policy[r, c]
        
        # 4) Step the environment
        next_state, reward, done = env.step(action)
        total_reward += reward

        if done:
            print("Episode finished with reward:", total_reward)
            break
        state = next_state

    # Save GA policy as a CSV file

    with open('ga_policy.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for s in range(MAX_STEPS + 1):
            writer.writerow([f"Step {s}"])
            for r in range(GRID_SIZE):
                writer.writerow(ga_policy[r, :, s])
            writer.writerow([])  # Blank line for separation
    env.close()
