import numpy as np
import tkinter as tk
from typing import Tuple, List, Dict
import random
import time

class GridEnvironment:
    def __init__(self, size: int = 10, max_steps: int = 50):
        self.size = size
        self.max_steps = max_steps
        self.current_steps = 0
        
        # Initialize grid (0: empty, 1: to seed)
        self.grid = np.zeros((size, size))
        
        # Initialize agent position (center of grid)
        self.start_pos = (size // 2, size // 2)
        self.agent_pos = self.start_pos
        
        # Initialize targets (example pattern - can be modified)
        self.targets = set([
            # (1, 3), (1, 4), (1, 5), (1, 6), (1, 8),
            # (2, 0), (2, 1), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
            # (3, 0), (3, 1), (3, 2), (3, 3), (3, 5), (3, 7),
            (4, 0), (4, 4),
            # (5, 1), (5, 5), (5, 6), (5, 7), (5, 9),
            (6, 5), (6, 8),
            # (7, 0), (7, 3), (7, 4), (7, 5), (7, 7), (7, 8), (7, 9),
            (8, 3), (8, 4), (8, 6), (8, 7),
            (9, 0), (9, 1), (9, 6),# (9, 7), (9, 9)
        ])
        for target in self.targets:
            self.grid[target] = 1
            
        self.visited_targets = set()
        
        # Define rewards
        self.REWARDS = {
            'step': -0.1,           # Small penalty for each step
            'visit_target': 1.0,    # Reward for visiting target cell
            'safe_return': 0.0,     # Reward for returning home safely
            'return_home': 5.0,     # Big reward for returning after visiting all targets
            'out_of_steps': -10.0   # Big penalty for running out of battery
        }
        
        # Initialize visualization
        self.root = None
        self.canvas = None
        self.cell_size = 40
        
    def reset(self) -> Tuple[Tuple[int, int], Dict]:
        """Reset the environment to initial state."""
        self.agent_pos = self.start_pos
        self.current_steps = 0
        self.visited_targets = set()
        return self.agent_pos, self._get_state_info()
        
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """Execute one step in the environment."""
        # Action mapping: 0: up, 1: right, 2: down, 3: left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Calculate new position
        new_pos = (
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        )
        
        # Check if move is valid
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.agent_pos = new_pos
            
        self.current_steps += 1
        
        # Calculate reward
        reward = self.REWARDS['step']
        done = False
        
        # Check if current position is a target
        if self.agent_pos in self.targets and self.agent_pos not in self.visited_targets:
            reward += self.REWARDS['visit_target']
            self.visited_targets.add(self.agent_pos)
            
        # Check if returned home
        if self.agent_pos == self.start_pos:
            if len(self.visited_targets) == len(self.targets):
                reward += self.REWARDS['return_home']
                done = True
            elif self.current_steps > 0:  # Only give safe return reward if some steps were taken
                reward += self.REWARDS['safe_return']
                done = True
                
        # Check if out of steps
        if self.current_steps >= self.max_steps:
            reward += self.REWARDS['out_of_steps']
            done = True
            
        return self.agent_pos, reward, done, self._get_state_info()
    
    def _get_state_info(self) -> Dict:
        """Return current state information."""
        return {
            'steps_remaining': self.max_steps - self.current_steps,
            'targets_remaining': len(self.targets) - len(self.visited_targets),
            'visited_targets': self.visited_targets,
            'grid': self.grid
        }
    
    def init_visualization(self):
        """Initialize Tkinter visualization."""
        self.root = tk.Tk()
        self.root.title("Grid Environment")
        
        canvas_size = self.size * self.cell_size
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size)
        self.canvas.pack()

        # Add labels for information
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=10)
        
        self.steps_label = tk.Label(self.info_frame, text="Steps: 0")
        self.steps_label.pack(side=tk.LEFT, padx=10)
        
        self.targets_label = tk.Label(self.info_frame, text="Targets: 0/0")
        self.targets_label.pack(side=tk.LEFT, padx=10)
        
        self.reward_label = tk.Label(self.info_frame, text="Total Reward: 0.0")
        self.reward_label.pack(side=tk.LEFT, padx=10)
        
    def render(self, total_reward=0):
        """Update visualization."""
        if self.root is None:
            self.init_visualization()
            
        self.canvas.delete("all")
        
        # Draw grid
        for i in range(self.size):
            for j in range(self.size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Draw cell
                color = "white"
                if (i, j) in self.targets:
                    color = "lightgreen" if (i, j) in self.visited_targets else "green"
                elif (i, j) == self.start_pos:
                    color = "yellow"
                    
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                
                # Draw agent
                if (i, j) == self.agent_pos:
                    self.canvas.create_oval(
                        x1 + 5, y1 + 5,
                        x2 - 5, y2 - 5,
                        fill="red"
                    )

        # Update labels
        self.steps_label.config(text=f"Steps: {self.current_steps}/{self.max_steps}")
        self.targets_label.config(text=f"Targets: {len(self.visited_targets)}/{len(self.targets)}")
        self.reward_label.config(text=f"Total Reward: {total_reward:.1f}")
                    
        self.root.update()
        
    def close(self):
        """Close visualization."""
        if self.root:
            self.root.destroy()
            self.root = None

if __name__ == "__main__":
    # Create environment
    env = GridEnvironment(size=10, max_steps=50)
    
    # Initialize visualization
    env.init_visualization()
    
    # Run random movements
    total_episodes = 5
    
    for episode in range(total_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        print(f"\nEpisode {episode + 1}")
        
        while not done:
            # Random action
            action = random.randint(0, 3)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Render
            env.render(total_reward)
            time.sleep(0.2)  # Add delay to make visualization visible
            
            # Print info
            print(f"Step: {env.current_steps}, Position: {env.agent_pos}, "
                  f"Reward: {reward:.1f}, Total Reward: {total_reward:.1f}")
            
            if done:
                print(f"Episode finished! Total Reward: {total_reward:.1f}")
                print(f"Visited {len(env.visited_targets)}/{len(env.targets)} targets")
                time.sleep(1)  # Pause at end of episode
                
    env.close()