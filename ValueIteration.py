import numpy as np
import numba
import pickle
import matplotlib.pyplot as plt

# Environment Setup
GRID_SIZE = 10
MAX_STEPS = 100
SAFE_RETURN_THRESHOLD = 0.3

# Grid setup
GRID = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.int32)

# Actions: Up, Right, Down, Left
ACTIONS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int32)
START_POS = (0, 0)
SEEDABLE_COUNT = np.sum(GRID == 1)

# Rewards
R_STEP = -0.01
R_NO_CELL = 0
R_REVISIT = 0
R_NEW_CELL = 2
R_BOUNDARY = 0
R_COMPLETION = 100.0
R_SAFE_RETURN = 0
R_BATTERY_DEPLETED = -10.0


@numba.njit
def step(r, c, visited, action, grid):
    """Simulate a step in the environment."""
    nr, nc = r + ACTIONS[action][0], c + ACTIONS[action][1]
    reward = R_STEP
    done = False
    
    # Check boundaries
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        return r, c, visited, R_BOUNDARY, done  # Stay in place, penalize

    # Copy visited to avoid modifying input directly
    new_visited = visited.copy()

    # Check cell state
    if grid[nr, nc] == 1:
        if new_visited[nr, nc]:
            reward += R_REVISIT
        else:
            reward += R_NEW_CELL
            new_visited[nr, nc] = 1  # Mark cell as visited
    else:
        reward += R_NO_CELL

    return nr, nc, new_visited, reward, done


@numba.njit
def value_iteration(grid, max_steps, gamma, theta):
    """Perform Value Iteration."""
    V = np.zeros((GRID_SIZE, GRID_SIZE, max_steps + 1), dtype=np.float32)
    policy = np.zeros((GRID_SIZE, GRID_SIZE, max_steps + 1), dtype=np.int32)

    for iteration in range(1000):  # Limit iterations to prevent infinite loops
        delta = 0.0
        new_V = V.copy()
        
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                for steps_remaining in range(max_steps + 1):
                    if steps_remaining == 0:
                        continue  # Skip terminal states
                    
                    best_value = -np.inf
                    best_action = 0

                    for action in range(4):  # Try all actions
                        nr, nc, _, reward, _ = step(r, c, np.zeros_like(grid), action, grid)
                        if steps_remaining > 1:
                            next_value = reward + gamma * V[nr, nc, steps_remaining - 1]
                        else:
                            next_value = reward
                        
                        if next_value > best_value:
                            best_value = next_value
                            best_action = action
                    
                    # Update value and policy
                    new_V[r, c, steps_remaining] = best_value
                    policy[r, c, steps_remaining] = best_action
                    
                    # Track max change
                    delta = max(delta, abs(V[r, c, steps_remaining] - best_value))
        
        V = new_V
        if delta < theta:
            break

    return V, policy


def main():
    gamma = 0.95  # Discount factor
    theta = 0.01  # Convergence threshold

    print("Starting Value Iteration...")
    V, policy = value_iteration(GRID, MAX_STEPS, gamma, theta)
    print("Value Iteration completed.")

    # Save results
    with open("value_iteration_policy.pkl", "wb") as f:
        pickle.dump((V, policy), f)

    # Plot value function for halfway through the steps
    plt.figure(figsize=(10, 5))
    plt.imshow(V[:, :, MAX_STEPS // 2], cmap="viridis")
    plt.colorbar(label="Value")
    plt.title(f"Value Function at {MAX_STEPS // 2} Steps Remaining")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


if __name__ == "__main__":
    main()
