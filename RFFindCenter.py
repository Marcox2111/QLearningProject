import pygame
import numpy as np
import random

class GridWorld:
    def __init__(self, width=400, height=400, grid_size=20):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("RL Agent Finding Center")
        
        # Grid parameters
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.rows = height // grid_size
        self.cols = width // grid_size
        
        # Center position
        self.center = (self.cols // 2, self.rows // 2)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # Agent parameters
        self.reset()
        
        # Q-learning parameters
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        
    def reset(self):
        # Random starting position
        self.agent_pos = (random.randint(0, self.cols-1), random.randint(0, self.rows-1))
        return self.agent_pos
    
    def get_state(self):
        return self.agent_pos
    
    def get_reward(self):
        if self.agent_pos == self.center:
            return 100
        return -1 - (abs(self.agent_pos[0] - self.center[0]) + abs(self.agent_pos[1] - self.center[1])) * 0.1
    
    def step(self, action):
        new_x = max(0, min(self.cols-1, self.agent_pos[0] + self.actions[action][0]))
        new_y = max(0, min(self.rows-1, self.agent_pos[1] + self.actions[action][1]))
        self.agent_pos = (new_x, new_y)
        
        reward = self.get_reward()
        done = (self.agent_pos == self.center)
        
        return self.agent_pos, reward, done
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        state_actions = self.q_table.get(state, [0, 0, 0, 0])
        return np.argmax(state_actions)
    
    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0, 0]
            
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value
    
    def draw(self):
        self.screen.fill(self.WHITE)
        
        # Draw grid
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, self.BLACK, (x, 0), (x, self.height))
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(self.screen, self.BLACK, (0, y), (self.width, y))
        
        # Draw center
        center_pixel = (self.center[0] * self.grid_size + self.grid_size//2,
                       self.center[1] * self.grid_size + self.grid_size//2)
        pygame.draw.circle(self.screen, self.GREEN, center_pixel, self.grid_size//3)
        
        # Draw agent
        agent_pixel = (self.agent_pos[0] * self.grid_size + self.grid_size//2,
                      self.agent_pos[1] * self.grid_size + self.grid_size//2)
        pygame.draw.circle(self.screen, self.RED, agent_pixel, self.grid_size//3)
        
        pygame.display.flip()

def main():
    env = GridWorld()
    clock = pygame.time.Clock()
    episodes = 1000
    max_steps = 100
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            action = env.get_action(state)
            next_state, reward, done = env.step(action)
            
            env.update_q_table(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            
            env.draw()
            clock.tick(100)  # Control animation speed
            
            if done:
                break
        
        # Decrease exploration rate
        env.epsilon = max(0.01, env.epsilon * 0.995)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {env.epsilon:.3f}")

if __name__ == "__main__":
    main()