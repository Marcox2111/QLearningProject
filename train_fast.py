import numpy as np
import numba
import pickle

# ------------------------------
# Environment Setup (No GUI)
# ------------------------------
GRID_SIZE = 10
MAX_STEPS = 60

# # Hardcoded grid from your code
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
# 
# GRID = np.array([
#     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ], dtype=np.int32)

# Rewards
R_STEP = -0.001
R_REVISIT_WALK = -0.0005
R_NO_CELL = 0
R_REVISIT = 0
R_NEW_CELL = 0.1
R_BOUNDARY = 0
R_RETURN_TOWARDS = 0
R_RETURN_AWAY = 0
R_COMPLETION = 2
R_SAFE_RETURN = 1
R_BATTERY_DEPLETED = 0
R_NOT_ENOUGH_SEEDS = 0.0
R_SLOW_RETURN = 0
R_FAST_RETURN = 0

# Actions: 0: up, 1: right, 2: down, 3: left
ACTIONS = np.array([(-1,0),(0,1),(1,0),(0,-1)], dtype=np.int32)

START_POS = (0,0)

# Precompute total seedable cells
SEEDABLE_COUNT = np.sum(GRID == 1)

@numba.njit
def get_epsilon(epsilon, decay_strategy, initial_epsilon, min_epsilon, decay_rate, decay_episode, effective_episodes):
    if decay_strategy == "exponential":
        epsilon=max(min_epsilon, initial_epsilon * (decay_rate ** decay_episode))
        # if epsilon < 0.05 and epsilon > min_epsilon:
        #     epsilon = 0.05
        # elif epsilon < min_epsilon:
        #     epsilon = 0
        return epsilon
    elif decay_strategy == "linear":
        linear_decay = (initial_epsilon - min_epsilon) / effective_episodes
        return max(min_epsilon, initial_epsilon - decay_episode * linear_decay)
    else:
        raise ValueError("Invalid decay strategy. Choose 'exponential' or 'linear'.")

@numba.njit
def step(r, c, visited, steps_used, seeded_count, action, grid, max_steps):
    # Apply action
    nr = r + ACTIONS[action][0]
    nc = c + ACTIONS[action][1]
    reward = R_STEP
    done = False
    
    # Check boundaries
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        nr = r
        nc = c
        reward += R_BOUNDARY
    
    if visited[nr, nc] == 1:
        reward += R_REVISIT_WALK

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
        reward += R_BATTERY_DEPLETED
        done = True
    elif nr == START_POS[0] and nc == START_POS[1]:
        if seeded_count == SEEDABLE_COUNT:
            reward += R_COMPLETION
            reward += R_FAST_RETURN * (max_steps - steps_used)
        else:
            reward += R_SAFE_RETURN
        done = True
    # if done:
    #     if nr == START_POS[0] and nc == START_POS[1]:
    #         if seeded_count == SEEDABLE_COUNT:
    #             reward += R_COMPLETION
    #             reward += R_FAST_RETURN * (max_steps - steps_used)
    #         else:
    #             reward += R_SAFE_RETURN
    #     else:
    #         reward += R_BATTERY_DEPLETED
    return nr, nc, visited, steps_used,seeded_count, reward, done

# State encoding/decoding
# We encode state as (r, c, steps_remaining), so indexing by steps_remaining directly.
# steps_remaining = max_steps - steps_used
# steps_used = max_steps - steps_remaining
# If we store Q in shape (GRID_SIZE, GRID_SIZE, MAX_STEPS+1, 4)
# Q[r, c, steps_remaining, action]
# steps_remaining = from 0 to MAX_STEPS

@numba.njit
def run_episode(Q, alpha, gamma, epsilon, grid, max_steps):
    # Initialize
    # visited is needed; we store it as a boolean mask (0/1)
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    r, c = START_POS
    visited[r, c] = 1
    steps_used = 0
    seeded_count = 0
    episode_reward = 0.0
    
    while True:
        steps_remaining = max_steps - steps_used
        # Epsilon-greedy
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[r, c, steps_remaining])
        
        nr, nc, visited, steps_used, seeded_count, reward, done = step(r, c, visited, steps_used,seeded_count, action, grid, max_steps)

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
    return episode_reward,epsilon

@numba.njit
def double_q_learning_update(Q1, Q2, r, c, steps_remaining, action, reward, nr, nc, next_steps_remaining, alpha, gamma, done):
    """
    Perform a Double Q-Learning update on Q1 and Q2.
    
    Parameters:
        Q1, Q2: Two Q-tables for Double Q-Learning (shape: GRID_SIZE x GRID_SIZE x MAX_STEPS+1 x 4)
        r, c: Current state (row, column)
        steps_remaining: Steps remaining in the episode
        action: Action taken
        reward: Reward received after taking the action
        nr, nc: Next state (row, column)
        next_steps_remaining: Steps remaining in the next state
        alpha: Learning rate
        gamma: Discount factor
        done: Boolean flag indicating whether the episode has terminated

    Returns:
        None (Q1 and Q2 are updated in place)
    """
    if np.random.random() < 0.5:
        # Update Q1
        if not done:
            max_action = np.argmax(Q1[nr, nc, next_steps_remaining])
            td_target = reward + gamma * Q2[nr, nc, next_steps_remaining, max_action]
        else:
            td_target = reward
        td_error = td_target - Q1[r, c, steps_remaining, action]
        Q1[r, c, steps_remaining, action] += alpha * td_error
    else:
        # Update Q2
        if not done:
            max_action = np.argmax(Q2[nr, nc, next_steps_remaining])
            td_target = reward + gamma * Q1[nr, nc, next_steps_remaining, max_action]
        else:
            td_target = reward
        td_error = td_target - Q2[r, c, steps_remaining, action]
        Q2[r, c, steps_remaining, action] += alpha * td_error

@numba.njit
def run_episode_double_q_learning(Q1, Q2, alpha, gamma, epsilon, grid, max_steps):
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    r, c = START_POS
    visited[r, c] = 1
    steps_used = 0
    seeded_count = 0
    episode_reward = 0.0
    
    while True:
        steps_remaining = max_steps - steps_used
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q1[r, c, steps_remaining] + Q2[r, c, steps_remaining])
        
        # Take action
        nr, nc, visited, steps_used, seeded_count, reward, done = step(r, c, visited, steps_used, seeded_count, action, grid, max_steps)
        
        # Calculate next steps remaining
        next_steps_remaining = max_steps - steps_used
        
        # Perform Double Q-Learning update
        double_q_learning_update(Q1, Q2, r, c, steps_remaining, action, reward, nr, nc, next_steps_remaining, alpha, gamma, done)
        
        # Accumulate reward
        episode_reward += reward
        
        # Terminate if done
        if done:
            break
        
        # Move to next state
        r, c = nr, nc
    
    return episode_reward


@numba.njit
def run_episode_extended_sarsa(Q, alpha, gamma, epsilon, grid, max_steps, n):
    # Initialize
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    r, c = START_POS
    visited[r, c] = 1
    steps_used = 0
    seeded_count = 0
    episode_reward = 0.0

    # Buffers to store the state, action, and reward for the last n steps
    state_buffer = []
    action_buffer = []
    reward_buffer = []

    while True:
        steps_remaining = max_steps - steps_used

        # Choose action using epsilon-greedy policy
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[r, c, steps_remaining])

        # Take action and observe reward and next state
        nr, nc, visited, steps_used,seeded_count, reward, done = step(r, c, visited, steps_used, seeded_count, action, grid, max_steps)
        
        # Store the current state, action, and reward in the buffers
        state_buffer.append((r, c, steps_remaining))
        action_buffer.append(action)
        reward_buffer.append(reward)

        # If we have at least n steps in the buffer, calculate the n-step return and update Q
        if len(state_buffer) >= n:
            # Calculate n-step return
            G = 0
            for i in range(n):
                G += (gamma ** i) * reward_buffer[i]
            
            # Add the bootstrapped Q-value for the nth state-action pair (if not done)
            if not done:
                next_state = state_buffer[n - 1]
                next_action = action_buffer[n - 1]
                G += (gamma ** n) * Q[next_state[0], next_state[1], next_state[2], next_action]
            
            # Update the Q-value for the first state-action pair in the buffer
            s, a = state_buffer[0], action_buffer[0]
            td_error = G - Q[s[0], s[1], s[2], a]
            Q[s[0], s[1], s[2], a] += alpha * td_error

            # Remove the first elements from the buffers (FIFO behavior)
            state_buffer.pop(0)
            action_buffer.pop(0)
            reward_buffer.pop(0)

        # Add reward to episode reward
        episode_reward += reward

        # Break if the episode is done
        if done:
            break

        # Move to the next state
        r, c = nr, nc

    # Handle remaining steps in the buffer (less than n steps left at the end)
    while len(state_buffer) > 0:
        # Calculate the return for the remaining steps
        G = 0
        for i in range(len(reward_buffer)):
            G += (gamma ** i) * reward_buffer[i]

        # Update Q for the first state-action pair in the buffer
        s, a = state_buffer[0], action_buffer[0]
        td_error = G - Q[s[0], s[1], s[2], a]
        Q[s[0], s[1], s[2], a] += alpha * td_error

        # Remove the first elements from the buffers
        state_buffer.pop(0)
        action_buffer.pop(0)
        reward_buffer.pop(0)

    return episode_reward


@numba.njit
def train_agent(n_episodes, alpha, gamma, initial_epsilon, min_epsilon, warmup_episodes, decay_rate, decay_strategy, grid, max_steps, convergence_window, min_improvement, patience, n_step=5):
    # Q = (np.random.rand(GRID_SIZE, GRID_SIZE, max_steps+1, 4) * 100).astype(np.float32)
    # Q = (np.ones((GRID_SIZE, GRID_SIZE, max_steps+1, 4)) * 1000).astype(np.float32)
    Q = np.full((GRID_SIZE, GRID_SIZE, max_steps+1, 4), 1e-5, dtype=np.float32)
    # Q= np.zeros((GRID_SIZE, GRID_SIZE, max_steps+1, 4), dtype=np.float32)
    Q2= np.zeros((GRID_SIZE, GRID_SIZE, max_steps+1, 4), dtype=np.float32)
    episode_rewards = np.zeros(n_episodes, dtype=np.float32)
    episode_epsilon = np.zeros(n_episodes, dtype=np.float32)
    # This marks the baseline episode from which we start decaying epsilon
    # Initially, we follow the original warmup logic.
    reset_base = warmup_episodes
    
    best_avg = -999999.0
    last_avg = -999999.0
    no_improvement_count = 0
    max_reward = -999999.0

    for ep in range(n_episodes):
        # Compute epsilon based on whether we are before or after reset_base
        if ep < reset_base:
            # Still in "warmup" or just reset phase, keep epsilon at initial_epsilon
            epsilon = initial_epsilon
        else:
            # Start decaying from the reset_base episode
            decay_episode = ep - reset_base
            # epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** decay_episode))
            epsilon = get_epsilon(initial_epsilon, decay_strategy, initial_epsilon, min_epsilon, decay_rate, decay_episode, n_episodes - reset_base)
        
        episode_epsilon[ep] = epsilon
        # Run an episode
        ep_reward,epsilon = run_episode(Q, alpha, gamma, epsilon, grid, max_steps)
        if ep_reward > max_reward:
            max_reward = ep_reward
        # ep_reward = run_episode_extended_sarsa(Q, alpha, gamma, epsilon, grid, max_steps, n_step)
        # ep_reward = run_episode_double_q_learning(Q, Q2, alpha, gamma, epsilon, grid, max_steps)
        episode_rewards[ep] = ep_reward
        good_episode=0
        good_reward=0
        if ep_reward > 1000:
            #safe this reward
            good_reward = ep_reward
            good_episode = ep
            print("good reward", good_reward, "good episode", good_episode)
        # Check improvement every convergence_window episodes
        if (ep + 1) % convergence_window == 0:
            current_window = episode_rewards[ep - convergence_window + 1 : ep + 1]
            current_avg = 0.0
            for v in current_window:
                current_avg += v
            current_avg /= convergence_window
            
            improvement = current_avg - last_avg

            # Update best average if improved
            if current_avg > best_avg:
                best_avg = current_avg
                print("Episode", ep+1, "avg reward:", float(current_avg), "improvement:", float(improvement), "epsilon:", epsilon, "max reward:", max_reward)
                

            # Check improvement thresholds
            if improvement < min_improvement:
                no_improvement_count += 1
                # print("No improvement for", no_improvement_count, "checks.")
            else:
                no_improvement_count = 0
            
            # If no improvement for 'patience' checks, reset epsilon decay
            if no_improvement_count >= patience:
                no_improvement_count = 0
                # Reset the baseline for epsilon decay to the current episode
                # This means we treat this point as a new start for decay,
                # so future episodes will have epsilon=initial and then decay again.
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
    warmup_episodes = n_episodes / 4
    effective_episodes = max(1, n_episodes - warmup_episodes)
    decay_rate = np.exp(np.log(min_epsilon / initial_epsilon) / (effective_episodes/1))
    decay_strategy = "exponential"
    steps = 5
    
    # Hyperparameters for improvement checking
    convergence_window = 1000
    min_improvement = 0.01
    patience = 1000
    
    Q, rewards, epsilons = train_agent(n_episodes, alpha, gamma, initial_epsilon, min_epsilon, warmup_episodes, decay_rate, decay_strategy, GRID, MAX_STEPS, convergence_window, min_improvement, patience,steps)
    # Plotting the rewards
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(range(len(rewards)), rewards, label="Episode Rewards")
    # plt.ylim(-10,0)
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Cumulative Reward")
    axs[0].set_title("Training Rewards Over Episodes")
    axs[0].grid()
    
    axs[1].plot(range(len(epsilons)), epsilons, label="Epsilon")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Epsilon")
    axs[1].set_title("Epsilon Decay Over Episodes")
    axs[1].grid()

    # plt.tight_layout()
    plt.show()
    # Save Q-table
    with open('trained_qtable.pkl', 'wb') as f:
        pickle.dump((Q, rewards), f)
    print("Training completed. Q-table saved.")
    

if __name__ == "__main__":
    main()
