import numpy as np
import numba
import pickle

# ------------------------------
# Environment Setup (No GUI)
# ------------------------------
GRID_SIZE = 10
MAX_STEPS = 70

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
R_STEP = -0.02
R_REVISIT_WALK = 0
R_NO_CELL = -0.02
R_REVISIT = -0.02
R_NEW_CELL = 0.1
R_BOUNDARY = -0.05
R_COMPLETION = 0
R_SAFE_RETURN = 0
R_BATTERY_DEPLETED = -0.3
R_LAZY = 0

ACTIONS = np.array([(-1,0),(0,1),(1,0),(0,-1)], dtype=np.int32)
START_POS = (0,0)
SEEDABLE_COUNT = np.sum(GRID == 1)

@numba.njit
def env_step(r, c, visited, steps_used, seeded_count, action):
    """One environment step for the drone."""
    nr = r + ACTIONS[action][0]
    nc = c + ACTIONS[action][1]
    reward = R_STEP
    done = False

    # Boundary check
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        nr = r
        nc = c
        reward += R_BOUNDARY
    
    if GRID[nr, nc] == 1:
        if visited[nr, nc] == 0:
            reward += R_NEW_CELL
            seeded_count += 1
        else:
            reward += R_REVISIT
    else:
        reward += R_NO_CELL
    
    visited[nr, nc] = 1

    steps_used += 1
    
    if steps_used > MAX_STEPS:
        reward += R_BATTERY_DEPLETED
        done = True
        
    if nr == START_POS[0] and nc == START_POS[1]:
        if seeded_count == SEEDABLE_COUNT:
            reward += R_COMPLETION
        else:
            reward += R_SAFE_RETURN
        if steps_used < MAX_STEPS/5:
            reward += R_LAZY
        done = True

    # if done:
    #     if nr == START_POS[0] and nc == START_POS[1]:
    #         if seeded_count == SEEDABLE_COUNT:
    #             reward += R_COMPLETION
    #         else:
    #             reward += R_SAFE_RETURN
    #     else:
    #         reward += R_BATTERY_DEPLETED

    return nr, nc, steps_used, seeded_count, reward, done

@numba.njit
def evaluate_policy(policy, debug=False): 
    """
    Runs exactly one episode using the given GA policy (shape = [GRID_SIZE, GRID_SIZE, MAX_STEPS+1]).
    Returns total reward (float).
    """
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    r, c = START_POS
    steps_used = 0
    seeded_count = 0
    total_reward = 0.0
    done = False


    while True:
        action = policy[r, c, steps_used]
        # action = policy[r, c]
        nr, nc, steps_used, seeded_count, reward, done = env_step(r, c, visited, steps_used, seeded_count, action)
        
        total_reward += reward
        
        if done:
            break
        r, c = nr, nc

    
    return total_reward

@numba.njit
def init_population(pop_size):
    """
    Create a population array of shape:
      (pop_size, GRID_SIZE, GRID_SIZE, MAX_STEPS+1)
    Each entry is a random action in [0..3].
    """
    population = np.zeros((pop_size, GRID_SIZE, GRID_SIZE, MAX_STEPS+1), dtype=np.int8)
    # population = np.zeros((pop_size, GRID_SIZE, GRID_SIZE), dtype=np.int8)
    for i in range(pop_size):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                for s in range(MAX_STEPS+1):
                    population[i, r, c, s] = np.random.randint(4)
                # population[i, r, c] = np.random.randint(4)
    return population

@numba.njit
def init_population_from_policy(pop_size, base_policy):
    """
    Create a population array initialized with mutations of the base policy.
    """
    grid_size_x, grid_size_y, base_max_steps = base_policy.shape  # Extract dimensions

    # Create population with the new max_steps
    population = np.empty((pop_size, grid_size_x, grid_size_y, MAX_STEPS + 1), dtype=np.int8)
    # Resize base policy to match new max_steps
    if base_max_steps < MAX_STEPS + 1:
        # If base_policy is smaller, pad with zeros
        new_base_policy = np.zeros((grid_size_x, grid_size_y, MAX_STEPS + 1), dtype=np.int8)
        new_base_policy[:, :, :base_max_steps] = base_policy
    else:
        # If base_policy is larger, truncate
        new_base_policy = base_policy[:, :, :MAX_STEPS + 1]
        
    # First member is exactly the base policy
    population[0] = new_base_policy.copy()
    
    # Rest of population will be mutated versions of base policy
    for i in range(1, pop_size):
        # Use higher mutation rate for initialization to ensure diversity
        population[i] = mutate(new_base_policy, mutation_rate=0.1)
    
    return population

@numba.njit
def mutate(policy, mutation_rate=0.001):
    """
    Mutate one policy by randomly changing some actions.
    We return a *new* policy array to keep it functional in JIT mode.
    """
    new_pol = np.copy(policy)
    n_states = GRID_SIZE * GRID_SIZE * (MAX_STEPS + 1)
    # n_states = GRID_SIZE * GRID_SIZE
    n_mutations = int(n_states * mutation_rate)
    for _ in range(n_mutations):
        rr = np.random.randint(GRID_SIZE)
        cc = np.random.randint(GRID_SIZE)
        ss = np.random.randint(MAX_STEPS+1)
        new_pol[rr, cc, ss] = np.random.randint(4)
        # new_pol[rr, cc] = np.random.randint(4)
    return new_pol

@numba.njit
def crossover(parent1, parent2):
    """
    Single-point crossover. Flatten both parents, pick a random cut,
    then combine slices.
    Returns a fresh child array with the same shape.
    """
    shape = parent1.shape
    size = parent1.size
    cut = np.random.randint(1, size)
    p1f = parent1.ravel()
    p2f = parent2.ravel()

    child_flat = np.empty(size, dtype=np.int8)
    child_flat[:cut] = p1f[:cut]
    child_flat[cut:] = p2f[cut:]

    return child_flat.reshape(shape)

@numba.njit
def run_generation(
    population,
    pop_size,
    elitism_count,
    mutation_rate):
    """
    Evaluates, sorts, reproduces the population **in one generation**.
    Returns:
      - new_population (updated for next gen)
      - best_idx (index in new_population of best individual)
      - best_fitness (float)
    """
    # Evaluate all
    fitnesses = np.empty(pop_size, dtype=np.float32)
    for i in range(pop_size):
        fitnesses[i] = evaluate_policy(population[i])

    # Sort by descending fitness
    # We'll do "argsort of negative" to get descending
    idx_sorted = np.argsort(-fitnesses)

    best_idx_in_generation = idx_sorted[0]
    best_fitness_in_generation = fitnesses[best_idx_in_generation]
    
    # Copy out the sorted population so we can do elitism & breeding
    # We'll store them in order
    sorted_pop = population[idx_sorted].copy()
    sorted_fit = fitnesses[idx_sorted].copy()

    # new_population to fill
    new_population = np.empty_like(population)

    # Elitism: keep top N as is
    for i in range(elitism_count):
        new_population[i] = sorted_pop[i]

    # Fill up the rest with crossover+mutation
    fill_index = elitism_count
    while fill_index < pop_size:
        # random pick two parents from top elitism_count
        p1_idx = np.random.randint(elitism_count)
        p2_idx = np.random.randint(elitism_count)
        parent1 = sorted_pop[p1_idx]
        parent2 = sorted_pop[p2_idx]

        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        new_population[fill_index] = child
        fill_index += 1

    # Return new population plus best info
    return new_population, best_idx_in_generation, best_fitness_in_generation, sorted_fit

def genetic_algorithm(
    pop_size=50, 
    elitism_ratio=0.2,
    mutation_rate=0.001,
    generations=50,
    starting_policy=None
):
    """
    Main GA loop, all JIT-compiled with no Python lists.
    """
    
    if starting_policy is not None:
        population = init_population_from_policy(pop_size, starting_policy)
        initial_fitness = evaluate_policy(starting_policy, False)
        best_global_fitness = initial_fitness
        best_global_policy = np.copy(starting_policy)
    else:
        # Initialize
        population = init_population(pop_size)
        best_global_fitness = -1e9
        best_global_policy = None
        
    n_elite = int(pop_size * elitism_ratio)
    if n_elite < 1:
        n_elite = 1


    for gen in range(generations):
        (population,
         best_idx_in_gen,
         best_fit_in_gen,
         sorted_fits) = run_generation(population, pop_size, n_elite, mutation_rate)

        # Track global best
        if best_fit_in_gen > best_global_fitness:
            best_global_fitness = best_fit_in_gen
            best_global_policy = np.copy(population[0])
            test= evaluate_policy(best_global_policy, False)
            # print("test best_global_policy", test)
            

        if (gen+1) % 10 == 0 or gen == generations-1:
            print(f"Generation {gen+1}/{generations}: Best fitness this gen = {best_fit_in_gen:.3f}, Overall best = {best_global_fitness:.3f}")

    return best_global_policy, best_global_fitness

def main():
    pop_size = 200
    elitism_ratio = 0.1
    mutation_rate = 0.2
    generations = 40000
    continue_from_last = True
    
    if continue_from_last:
        with open('best_ga_policy.pkl', 'rb') as f:
            best_pol = pickle.load(f)
    
    try:
        best_pol, best_fit = genetic_algorithm(
            pop_size=pop_size,
            elitism_ratio=elitism_ratio,
            mutation_rate=mutation_rate,
            generations=generations,
            starting_policy=best_pol
        )
    except:
        best_pol, best_fit = genetic_algorithm(
            pop_size=pop_size,
            elitism_ratio=elitism_ratio,
            mutation_rate=mutation_rate,
            generations=generations
        )
    print("GA complete. Best fitness achieved:", best_fit)
    final_check = evaluate_policy(best_pol,False)
    print("Check immediate su best_pol, reward =", final_check)


    # Save best policy for later
    with open('best_ga_policy.pkl', 'wb') as f:
        pickle.dump(best_pol, f)
    
    with open('best_ga_policy.pkl', 'rb') as f:
        best_pol = pickle.load(f)
    check = evaluate_policy(best_pol, False)
    print("Check after loading su best_pol, reward =", check)

    print("Best policy saved to best_ga_policy.pkl")

if __name__ == "__main__":
    main()
