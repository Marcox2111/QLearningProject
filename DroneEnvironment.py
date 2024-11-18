import numpy as np
import tkinter as tk
import time
from tkinter import ttk
import math

class DroneEnvironment:
    def __init__(self, grid_size=10, max_steps=100, render_mode=None, safe_return_threshold=0.3):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.safe_return_threshold = safe_return_threshold
        
        # Colors for the GUI
        self.COLORS = {
            'unseeded': '#FFFFFF',    # White
            'seedable': '#48C4A4',    # Light blue
            'seeded': '#57B9BF',      # Blue
            'start': '#FF6347',       # Tomato red
            'drone': '#2980B9',       # Dark blue
            'drone_return': '#F39C12'  # Orange
        }
        
        # Action space: 0: up, 1: right, 2: down, 3: left
        self.num_actions = 4
        
        # Initialize GUI if render_mode is "human"
        if self.render_mode == "human":
            self._init_gui()


        self.REWARDS = {
            'step': -0.01,             # Small penalty to encourage efficiency
            'boundary': -0.05,         # Moderate penalty for poor navigation
            'new_cell': 0.1,           # Good reward for doing the main task
            'return_towards': 0.05,    # Encourage safe return behavior
            'return_away': -0.1,       # Stronger penalty for risky behavior
            'completion': 4.0,         # Big reward for perfect mission
            'safe_return': 0.3,        # Good reward for safe return
            'battery_depleted': -2.0   # Catastrophic failure - should be strongly avoided
        }
        
        self.reset()

    def _init_gui(self):
        """Initialize Tkinter GUI"""
        self.root = tk.Tk()
        self.root.title("Drone Seeding Environment")
        
        # Calculate cell size based on screen size
        screen_width = self.root.winfo_screenwidth() * 0.5  # Use 50% of screen width
        self.cell_size = min(50, int(screen_width / self.grid_size))
        canvas_size = self.cell_size * self.grid_size
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.main_frame, 
            width=canvas_size, 
            height=canvas_size + 30,  # Extra space for battery
            background='white'
        )
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize update flag
        self.needs_update = True
        
    def reset(self):
        # Generate random grid
        self.grid = np.random.choice([0, 1], size=(self.grid_size, self.grid_size), p=[0.5, 0.5])
        
        # Random starting position
        valid_positions = np.argwhere(self.grid == 1)
        if len(valid_positions) == 0:
            self.grid[0, 0] = 1
            valid_positions = np.array([[0, 0]])
        start_idx = np.random.randint(len(valid_positions))
        self.start_pos = valid_positions[start_idx]
        self.current_pos = self.start_pos.copy()
        
        # Initialize visited cells
        self.visited = np.zeros((self.grid_size, self.grid_size))
        self.visited[tuple(self.current_pos)] = 1
        
        # Reset steps
        self.steps = 0
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self.get_state()

    def get_state(self):
        return (
            tuple(self.current_pos),
            self.max_steps - self.steps,
            tuple(map(tuple, self.visited))
        )

    def _calculate_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _should_return_home(self):
        remaining_battery = self.max_steps - self.steps
        distance_to_start = self._calculate_manhattan_distance(self.current_pos, self.start_pos)
        safe_distance = distance_to_start + 2
        return remaining_battery <= (safe_distance + self.max_steps * self.safe_return_threshold)

    def step(self, action):
        self.steps += 1
        
        # Movement directions
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        
        # Calculate new position
        new_pos = self.current_pos + directions[action]
        
        # Initialize reward and termination flag
        terminated = False
        reward = self.REWARDS['step']
        
        # Check boundaries
        if (new_pos[0] < 0 or new_pos[0] >= self.grid_size or 
            new_pos[1] < 0 or new_pos[1] >= self.grid_size):
            new_pos = self.current_pos  # Stay in current position
            reward += self.REWARDS['boundary']
        
        # Update position and mark as visited
        self.current_pos = new_pos
        if not self.visited[tuple(self.current_pos)]:
            self.visited[tuple(self.current_pos)] = 1
            if self.grid[tuple(self.current_pos)] == 1:
                reward += self.REWARDS['new_cell']

        # Check if drone should return home
        should_return = self._should_return_home()
        if should_return:
            distance_to_start = self._calculate_manhattan_distance(self.current_pos, self.start_pos)
            new_distance = self._calculate_manhattan_distance(new_pos, self.start_pos)
            if new_distance < distance_to_start:
                reward += self.REWARDS['return_towards']
            else:
                reward += self.REWARDS['return_away']

        # Check termination conditions
        if self.steps >= self.max_steps:
            reward += self.REWARDS['battery_depleted']
            terminated = True
        elif np.array_equal(self.current_pos, self.start_pos):
            if np.all(self.visited[self.grid == 1] == 1):
                reward += self.REWARDS['completion']
            else:
                reward += self.REWARDS['safe_return']
            terminated = True

        if self.render_mode == "human":
            self._render_frame()
        
        return self.get_state(), reward, terminated

    def _render_frame(self):
        """Render the current state in the GUI"""
        if not hasattr(self, 'canvas'):
            return
            
        self.canvas.delete("all")  # Clear canvas
        
        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Choose color based on cell state
                if self.grid[i, j] == 0:
                    color = self.COLORS['unseeded']
                else:
                    if self.visited[i, j] == 1:
                        color = self.COLORS['seeded']
                    else:
                        color = self.COLORS['seedable']
                
                # Draw cell
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='gray')
        
        # Draw start position
        start_x = self.start_pos[1] * self.cell_size + self.cell_size/2
        start_y = self.start_pos[0] * self.cell_size + self.cell_size/2
        self.canvas.create_oval(
            start_x - self.cell_size/4, start_y - self.cell_size/4,
            start_x + self.cell_size/4, start_y + self.cell_size/4,
            fill=self.COLORS['start']
        )
        
        # Draw drone
        drone_x = self.current_pos[1] * self.cell_size + self.cell_size/2
        drone_y = self.current_pos[0] * self.cell_size + self.cell_size/2
        drone_color = self.COLORS['drone_return'] if self._should_return_home() else self.COLORS['drone']
        self.canvas.create_oval(
            drone_x - self.cell_size/3, drone_y - self.cell_size/3,
            drone_x + self.cell_size/3, drone_y + self.cell_size/3,
            fill=drone_color
        )
        
        # Draw battery bar
        battery_y = self.grid_size * self.cell_size + 15
        battery_width = self.grid_size * self.cell_size - 20
        self.canvas.create_rectangle(10, battery_y, battery_width + 10, battery_y + 10, outline='gray')
        
        # Battery level
        remaining = (self.max_steps - self.steps) / self.max_steps
        if remaining > 0:
            # Choose color based on battery level
            if remaining > 0.6:
                color = '#2ECC71'  # Green
            elif remaining > 0.3:
                color = '#F1C40F'  # Yellow
            else:
                color = '#E74C3C'  # Red
                
            self.canvas.create_rectangle(
                10, battery_y,
                10 + battery_width * remaining, battery_y + 10,
                fill=color, outline=''
            )
        
        # Draw return threshold line
        threshold_x = 10 + battery_width * self.safe_return_threshold
        self.canvas.create_line(
            threshold_x, battery_y - 5,
            threshold_x, battery_y + 15,
            fill='#E74C3C', width=2
        )
        
        # Update the display
        self.root.update()
        time.sleep(0.2)  # Add small delay for visualization

    def close(self):
        """Close the GUI"""
        if hasattr(self, 'root'):
            self.root.destroy()
            del self.root

def test_environment():
    """Test function to check if the environment works correctly"""
    try:
        # Create environment
        env = DroneEnvironment(grid_size=5, max_steps=50, render_mode="human")
        
        # Run a few test steps
        state = env.reset()
        time.sleep(1)
        
        # Run some random actions
        for _ in range(20):
            action = np.random.randint(4)
            state, reward, done = env.step(action)
            
            if done:
                print("Episode finished")
                break
        
        time.sleep(2)
        env.close()
        
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test_environment()