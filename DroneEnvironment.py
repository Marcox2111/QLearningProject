import pickle
import numpy as np
import tkinter as tk
import time
from tkinter import ttk

GRID_SIZE = 10
MAX_STEPS = 60
SAFE_RETURN_THRESHOLD = 0.3

# Action definitions
ACTIONS = np.array([(-1,0),(0,1),(1,0),(0,-1)], dtype=np.int32)

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



# # Hardcoded grid from your training code
GRID = np.array([
    [0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
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

SEEDABLE_COUNT = np.sum(GRID == 1)
START_POS = (0,0)

def manhattan_distance(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)

def should_return_home(r, c, steps_used, max_steps, safe_return_thr):
    remaining = max_steps - steps_used
    dist = manhattan_distance(r, c, START_POS[0], START_POS[1])
    safe_distance = dist + 2
    return remaining <= (safe_distance + max_steps * safe_return_thr)

class DroneEnvironment:
    def __init__(self, grid=GRID, grid_size=GRID_SIZE, max_steps=MAX_STEPS, render_mode="human", safe_return_threshold=SAFE_RETURN_THRESHOLD):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.safe_return_threshold = safe_return_threshold
        self.grid = grid
        self.i=0

        # In training code, start is always (0,0)
        self.start_pos = np.array([0,0], dtype=np.int32)

        # Colors for visualization
        self.COLORS = {
            'unseeded': '#FFFFFF',
            'seedable': '#48C4A4',
            'seeded': '#57B9BF',
            'start': '#FF6347',
            'drone': '#2980B9',
            'drone_return': '#F39C12'
        }

        # Initialize GUI if human mode
        if self.render_mode == "human":
            self._init_gui()
        
        self.reset()

    def _init_gui(self):
        self.root = tk.Tk()
        self.root.title("Drone Seeding Environment")

        screen_width = self.root.winfo_screenwidth() * 0.5
        self.cell_size = min(50, int(screen_width / self.grid_size))
        canvas_size = self.cell_size * self.grid_size

        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.canvas = tk.Canvas(
            self.main_frame,
            width=canvas_size,
            height=canvas_size + 30,
            background='white'
        )
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.update()

    def reset(self):
        self.current_pos = np.array(self.start_pos, dtype=np.int32)
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.visited[self.current_pos[0], self.current_pos[1]] = 1
        self.steps_used = 0

        if self.render_mode == "human":
            self._render_frame()

        return (tuple(self.current_pos), self.max_steps - self.steps_used)

    def step(self, action):
        r, c = self.current_pos
        nr = r + ACTIONS[action][0]
        nc = c + ACTIONS[action][1]
        reward = R_STEP
        done = False

        # Check boundaries
        if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size:
            nr = r
            nc = c
            reward += R_BOUNDARY

        if self.grid[nr, nc] == 1:
            if self.visited[nr, nc] == 0:
                reward += R_NEW_CELL
            else:
                reward += R_REVISIT
        else:
            reward += R_NO_CELL

        self.visited[nr, nc] = 1
        
        self.current_pos = np.array([nr, nc], dtype=np.int32)


        # Termination conditions
        self.steps_used += 1
        if self.steps_used > self.max_steps:
            reward += R_BATTERY_DEPLETED
            done = True
            
        # if done:
            # if nr == START_POS[0] and nc == START_POS[1]:
            #     seeded_count = 0
            #     for i in range(self.grid_size):
            #         for j in range(self.grid_size):
            #             if self.grid[i, j] == 1 and self.visited[i, j] == 1:
            #                 seeded_count += 1
            #     if seeded_count == SEEDABLE_COUNT:
            #         reward += R_COMPLETION
            #     else:
            #         reward += R_SAFE_RETURN
            # else:
            #     reward += R_BATTERY_DEPLETED
            
        if nr == START_POS[0] and nc == START_POS[1]:
            seeded_count = 0
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.grid[i, j] == 1 and self.visited[i, j] == 1:
                        seeded_count += 1
            if seeded_count == SEEDABLE_COUNT:
                reward += R_COMPLETION
            else:
                reward += R_SAFE_RETURN
            done = True
            

        if self.render_mode == "human":
            self._render_frame()

        return (tuple(self.current_pos), self.max_steps - self.steps_used), reward, done

    def close(self):
        if hasattr(self, 'root'):
            self.root.destroy()

    def _should_return_home(self):
        return should_return_home(
            self.current_pos[0],
            self.current_pos[1],
            self.steps_used,
            self.max_steps,
            self.safe_return_threshold
        )
        
    def _resume_rendering(self, event=None):
        """ Callback to resume rendering when a key is pressed. """
        self.waiting_for_key.set(True)
        self.root.unbind("<space>")  # Unbind after first key press
    
    def _render_frame(self):
        if not hasattr(self, 'canvas'):
            return

        if self.i == 0:
            self.waiting_for_key = tk.BooleanVar(value=False)
            self.root.bind("<space>", self._resume_rendering)  # Bind key event
            self.root.wait_variable(self.waiting_for_key)  # Wait until key is pressed
        self.i += 1
        self.canvas.delete("all")

        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if self.grid[i, j] == 1:
                    if self.visited[i, j] == 1:
                        color = self.COLORS['seeded']
                    else:
                        color = self.COLORS['seedable']
                else:
                    color = self.COLORS['unseeded']

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

        remaining = (self.max_steps - self.steps_used) / self.max_steps
        if remaining > 0:
            if remaining > 0.6:
                color = '#2ECC71'
            elif remaining > 0.3:
                color = '#F1C40F'
            else:
                color = '#E74C3C'

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

        steps_remaining = self.max_steps - self.steps_used
        self.canvas.create_text(
            5, battery_y + 5,
            anchor='w',
            text=f"Steps Left: {steps_remaining}",
            fill='black',
            font=('Helvetica', 12, 'bold')
        )

        self.root.update()
        time.sleep(0.1)  # small delay for visualization

if __name__ == "__main__":
    # Example on how to load Q-table and visualize a single episode
    # Make sure you have 'trained_qtable.pkl' from your training run.
    with open('trained_qtable.pkl', 'rb') as f:
        Q, rewards = pickle.load(f)

    env = DroneEnvironment(render_mode="human")
    state = env.reset()
    total_reward = 0.0

    while True:
        (r, c), steps_remaining = state
        # Choose best action from Q-table
        action = np.argmax(Q[r, c, steps_remaining])
        next_state, reward, done = env.step(action)
        total_reward += reward
        if done:
            print("Episode finished with reward:", total_reward)
            break
        state = next_state

    env.close()
