import numpy as np
import tkinter as tk
import random
import time
from itertools import product

# Define rewards globally
R_STEP = -0.01
R_SAFE_RETURN = 0.5
R_NEW_CELL = 3.0
R_COMPLETION = 10.0
R_BATTERY_DEPLETED = -10.0

class GridWorld:
    def __init__(self, size=10, num_ones=4):
        self.size = size
        self.grid = np.zeros((size, size))
        
        # Place 1s randomly
        positions = [(i, j) for i in range(size) for j in range(size)]
        one_positions = random.sample(positions, num_ones)
        for pos in one_positions:
            self.grid[pos] = 1
            
        self.initial_pos = (size-1, size-1)
        self.target_positions = [(i, j) for i in range(size) for j in range(size) if self.grid[i, j] == 1]
        
    def get_valid_actions(self, state):
        x, y = state[0]
        actions = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                actions.append((dx, dy))
        return actions

def value_iteration(grid_world, max_steps=100, gamma=0.99, theta=0.001):
    positions = [(i, j) for i in range(grid_world.size) for j in range(grid_world.size)]
    target_states = list(product([0, 1], repeat=len(grid_world.target_positions)))
    steps_remaining = list(range(max_steps + 1))
    
    V = {}
    policy = {}
    for pos in positions:
        for target_state in target_states:
            for steps in steps_remaining:
                V[((pos), tuple(target_state), steps)] = 0
                policy[((pos), tuple(target_state), steps)] = (0, 0)
    
    while True:
        delta = 0
        for pos in positions:
            for target_state in target_states:
                for steps in steps_remaining:
                    if steps == 0:  # No steps remaining
                        if pos == grid_world.initial_pos:
                            V[((pos), tuple(target_state), steps)] = R_SAFE_RETURN
                            if all(target_state):
                                V[((pos), tuple(target_state), steps)] += R_COMPLETION
                        else:
                            V[((pos), tuple(target_state), steps)] = R_BATTERY_DEPLETED
                        continue
                        
                    state = (pos, tuple(target_state), steps)
                    v = V[state]
                    values = []
                    actions = grid_world.get_valid_actions(state)
                    
                    for action in actions:
                        next_pos = (pos[0] + action[0], pos[1] + action[1])
                        next_target_state = list(target_state)
                        reward = R_STEP  # Base step cost
                        
                        # Check if we're visiting a new target
                        if (next_pos in grid_world.target_positions and 
                            next_target_state[grid_world.target_positions.index(next_pos)] == 0):
                            reward += R_NEW_CELL
                            next_target_state[grid_world.target_positions.index(next_pos)] = 1
                            
                        # Check if we're returning to start
                        if next_pos == grid_world.initial_pos:
                            reward += R_SAFE_RETURN
                            if all(next_target_state):  # All targets collected
                                reward += R_COMPLETION
                        
                        next_state = (next_pos, tuple(next_target_state), steps - 1)
                        values.append(reward + gamma * V[next_state])
                    
                    if values:
                        best_value = max(values)
                        V[state] = best_value
                        best_action = actions[values.index(best_value)]
                        policy[state] = best_action
                        
                        delta = max(delta, abs(v - V[state]))
        
        if delta < theta:
            break
            
    return V, policy

def find_path(grid_world, policy, max_steps=100):
    path = [grid_world.initial_pos]
    current_pos = grid_world.initial_pos
    visited_targets = [0] * len(grid_world.target_positions)
    steps_remaining = max_steps
    
    while steps_remaining > 0:
        state = (current_pos, tuple(visited_targets), steps_remaining)
        action = policy[state]
        next_pos = (int(current_pos[0] + action[0]), int(current_pos[1] + action[1]))
        
        if next_pos in grid_world.target_positions:
            idx = grid_world.target_positions.index(next_pos)
            visited_targets[idx] = 1
            
        path.append(next_pos)
        current_pos = next_pos
        steps_remaining -= 1
        
        # If at initial position and (all targets collected or low on steps), end
        if current_pos == grid_world.initial_pos and (all(visited_targets) or steps_remaining <= len(path)):
            break
            
    return path

class GridWorldGUI:
    def __init__(self, grid_world, path, max_steps=100):
        self.root = tk.Tk()
        self.root.title("GridWorld Path Visualization")
        self.cell_size = 50
        self.grid_world = grid_world
        self.path = path
        self.current_step = 0
        self.visited_targets = set()
        self.max_steps = max_steps
        
        canvas_size = self.cell_size * grid_world.size
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size + 30)
        self.canvas.pack()
        
        self.initial_cell_color = "lightblue"
        self.draw_grid()
        self.animate()
        
    def draw_grid(self):
        # Draw step counter
        self.canvas.delete("counter")
        self.canvas.create_text(
            self.cell_size * self.grid_world.size // 2,
            self.cell_size * self.grid_world.size + 15,
            text=f"Steps: {self.current_step}/{self.max_steps}",
            tags="counter"
        )
        
        for i in range(self.grid_world.size):
            for j in range(self.grid_world.size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                if (i, j) == self.grid_world.initial_pos:
                    color = self.initial_cell_color
                elif (i, j) in self.visited_targets:
                    color = "green"
                elif self.grid_world.grid[i, j] == 1:
                    color = "yellow"
                else:
                    color = "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                
    def animate(self):
        if self.current_step < len(self.path):
            self.canvas.delete("agent")
            i, j = self.path[self.current_step]
            
            if self.grid_world.grid[i, j] == 1:
                self.visited_targets.add((i, j))
            
            self.draw_grid()
            
            x = j * self.cell_size + self.cell_size//2
            y = i * self.cell_size + self.cell_size//2
            self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="red", tags="agent")
            
            self.current_step += 1
            self.root.after(70, self.animate)
            
    def run(self):
        self.root.mainloop()

def main():
    max_steps = 30  # Reduced steps to make it more challenging
    grid_world = GridWorld()
    V, policy = value_iteration(grid_world, max_steps=max_steps)
    path = find_path(grid_world, policy, max_steps=max_steps)
    gui = GridWorldGUI(grid_world, path, max_steps=max_steps)
    gui.run()

if __name__ == "__main__":
    main()