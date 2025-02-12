import numpy as np
import numba
import pickle

# ------------------------------
# Environment Setup (No GUI)
# ------------------------------
GRID_SIZE = 10
MAX_STEPS = 60

GRID = np.array([
    [0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
], dtype=np.int32)

# Rewards
R_STEP = -0.02
R_REVISIT_WALK = 0
R_NO_CELL = -0.02   
R_REVISIT = -0.02
R_NEW_CELL = 0
R_BOUNDARY = -0.05
R_COMPLETION = 0
R_SAFE_RETURN = 0
R_BATTERY_DEPLETED = -0.3

ACTIONS = np.array([(-1,0),(0,1),(1,0),(0,-1)], dtype=np.int32)
START_POS = (0,0)
SEEDABLE_COUNT = np.sum(GRID == 1)

@numba.njit
def get_epsilon(epsilon, decay_strategy, initial_epsilon, min_epsilon, decay_rate, decay_episode, effective_episodes):
    if decay_strategy == "exponential":
        epsilon=max(min_epsilon, initial_epsilon * (decay_rate ** decay_episode))
        # if epsilon < 0.05 and epsilon > min_epsilon:
        #     epsilon = 0.05
        if epsilon < min_epsilon:
            epsilon = 0
        return epsilon
    elif decay_strategy == "linear":
        linear_decay = (initial_epsilon - min_epsilon) / effective_episodes
        return max(min_epsilon, initial_epsilon - decay_episode * linear_decay)
    else:
        raise ValueError("Invalid decay strategy. Choose 'exponential' or 'linear'.")

@numba.njit
def step(r, c, visited, steps_used, seeded_count, action, grid, max_steps):
    nr = r + ACTIONS[action][0]
    nc = c + ACTIONS[action][1]
    reward = R_STEP
    done = False
    
    # Check boundaries
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        nr = r
        nc = c
        reward += R_BOUNDARY
    
    if grid[nr, nc] == 1:
        if visited[nr, nc] == 1:
            reward += R_REVISIT
        else:
            reward += R_NEW_CELL
            seeded_count += 1
    elif grid[nr, nc] == 0:
        reward += R_NO_CELL

    visited[nr, nc] = 1
    
    # Termination conditions
    steps_used += 1
    if steps_used >= max_steps:
        done = True
    if done:
        if nr == START_POS[0] and nc == START_POS[1]:
            if seeded_count == SEEDABLE_COUNT:
                reward += R_COMPLETION
            else:
                reward += R_SAFE_RETURN
        else:
            reward += R_BATTERY_DEPLETED

    return nr, nc, visited, steps_used, seeded_count, reward, done

@numba.njit
def run_episode(Q, alpha, gamma, epsilon, grid, max_steps):
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    r, c = START_POS
    visited[r, c] = 1
    steps_used = 0
    seeded_count = 0
    episode_reward = 0.0
    
    while True:
        steps_remaining = max_steps - steps_used
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[r, c, steps_remaining])
        
        nr, nc, visited, steps_used, seeded_count, reward, done = step(
            r, c, visited, steps_used, seeded_count, action, grid, max_steps)

        next_steps_remaining = max_steps - steps_used
        if not done:
            td_target = reward + gamma * np.max(Q[nr, nc, next_steps_remaining])
        else:
            td_target = reward
        
        td_error = td_target - Q[r, c, steps_remaining, action]
        Q[r, c, steps_remaining, action] += alpha * td_error
        
        episode_reward += reward
        if done:
            break
        r, c = nr, nc
    return episode_reward, epsilon

@numba.njit
def train_agent(n_episodes, alpha, gamma, initial_epsilon, min_epsilon, 
                warmup_episodes, decay_rate, decay_strategy, grid, max_steps, 
                convergence_window, min_improvement, patience):
    Q = (np.ones((GRID_SIZE, GRID_SIZE, max_steps+1, 4)) * 1000).astype(np.float32)
    episode_rewards = np.zeros(n_episodes, dtype=np.float32)
    episode_epsilon = np.zeros(n_episodes, dtype=np.float32)
    reset_base = warmup_episodes
    
    best_avg = -999999.0
    last_avg = -999999.0
    no_improvement_count = 0
    max_reward = -999999.0

    for ep in range(n_episodes):
        if ep < reset_base:
            epsilon = initial_epsilon
        else:
            decay_episode = ep - reset_base
            epsilon = get_epsilon(initial_epsilon, decay_strategy, initial_epsilon, 
                                min_epsilon, decay_rate, decay_episode, 
                                n_episodes - reset_base)
        
        episode_epsilon[ep] = epsilon
        ep_reward, epsilon = run_episode(Q, alpha, gamma, epsilon, grid, max_steps)
        if ep_reward > max_reward:
            max_reward = ep_reward
        episode_rewards[ep] = ep_reward

        if (ep + 1) % convergence_window == 0:
            current_window = episode_rewards[ep - convergence_window + 1 : ep + 1]
            current_avg = 0.0
            for v in current_window:
                current_avg += v
            current_avg /= convergence_window
            
            improvement = current_avg - last_avg

            if current_avg > best_avg:
                best_avg = current_avg
                print("Episode", ep+1, "avg reward:", float(current_avg), 
                      "improvement:", float(improvement), "epsilon:", epsilon, 
                      "max reward:", max_reward)

            if improvement < min_improvement:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            
            if no_improvement_count >= patience:
                no_improvement_count = 0
                reset_base = ep
                effective_episodes = max(1, n_episodes - reset_base)
                decay_rate = np.exp(np.log(min_epsilon / initial_epsilon) / (effective_episodes/1))
                print("No improvement for", patience, "checks. Resetting epsilon baseline at episode", ep+1)
            
            last_avg = current_avg

    return Q, episode_rewards, episode_epsilon

def main():
    n_episodes = 1000000
    alpha = 0.1
    gamma = 0.95
    initial_epsilon = 1
    min_epsilon = 0.001
    warmup_episodes = n_episodes // 5
    effective_episodes = max(1, n_episodes - warmup_episodes)
    decay_rate = np.exp(np.log(min_epsilon / initial_epsilon) / (effective_episodes/1))
    decay_strategy = "exponential"
    
    convergence_window = 1000
    min_improvement = 0.01
    patience = 1000
    
    Q, rewards, epsilons = train_agent(n_episodes, alpha, gamma, initial_epsilon, 
                                     min_epsilon, warmup_episodes, decay_rate, 
                                     decay_strategy, GRID, MAX_STEPS, 
                                     convergence_window, min_improvement, patience)
    
    # Plotting
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(range(len(rewards)), rewards, label="Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Cumulative Reward")
    axs[0].set_title("Training Rewards Over Episodes")
    axs[0].grid()
    
    axs[1].plot(range(len(epsilons)), epsilons, label="Epsilon")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Epsilon")
    axs[1].set_title("Epsilon Decay Over Episodes")
    axs[1].grid()

    plt.show()
    
    with open('trained_qtable.pkl', 'wb') as f:
        pickle.dump((Q, rewards), f)
    print("Training completed. Q-table saved.")

if __name__ == "__main__":
    main()