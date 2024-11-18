import pygame
import numpy as np
import random
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Agent:
    position: Tuple[int, int]
    color: Tuple[int, int, int]
    total_reward: float = 0
    done: bool = False

class MultiAgentGridWorld:
    def __init__(self, width=600, height=600, grid_size=20, num_agents=10):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Multi-Agent RL Finding Center")
        
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
        self.GREEN = (0, 255, 0)
        
        # Multi-agent setup
        self.num_agents = num_agents
        self.generate_agent_colors()
        self.reset()
        
        # Q-learning parameters
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        
        # Statistics
        self.episode_count = 0
        self.success_count = 0
    
    def generate_agent_colors(self):
        # Generate distinct colors for each agent
        self.agent_colors = []
        for i in range(self.num_agents):
            hue = i / self.num_agents
            # Convert HSV to RGB (simple version)
            h = hue * 6
            c = 255
            x = int(c * (1 - abs(h % 2 - 1)))
            
            if h < 1: rgb = (c, x, 0)
            elif h < 2: rgb = (x, c, 0)
            elif h < 3: rgb = (0, c, x)
            elif h < 4: rgb = (0, x, c)
            elif h < 5: rgb = (x, 0, c)
            else: rgb = (c, 0, x)
            
            self.agent_colors.append(rgb)
    
    def reset(self):
        # Initialize multiple agents with random positions
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                position=(random.randint(0, self.cols-1), random.randint(0, self.rows-1)),
                color=self.agent_colors[i]
            )
            self.agents.append(agent)
        return [agent.position for agent in self.agents]
    
    def get_reward(self, position):
        if position == self.center:
            return 100
        return -1 - (abs(position[0] - self.center[0]) + abs(position[1] - self.center[1])) * 0.1
    
    def step(self, agent_idx, action):
        agent = self.agents[agent_idx]
        if agent.done:
            return agent.position, 0, True
        
        new_x = max(0, min(self.cols-1, agent.position[0] + self.actions[action][0]))
        new_y = max(0, min(self.rows-1, agent.position[1] + self.actions[action][1]))
        agent.position = (new_x, new_y)
        
        reward = self.get_reward(agent.position)
        agent.total_reward += reward
        agent.done = (agent.position == self.center)
        
        if agent.done:
            self.success_count += 1
        
        return agent.position, reward, agent.done
    
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
        
        # Draw agents
        for agent in self.agents:
            agent_pixel = (agent.position[0] * self.grid_size + self.grid_size//2,
                         agent.position[1] * self.grid_size + self.grid_size//2)
            pygame.draw.circle(self.screen, agent.color, agent_pixel, self.grid_size//3)
        
        # Draw statistics
        font = pygame.font.Font(None, 36)
        stats_text = f"Episode: {self.episode_count} | Success Rate: {self.success_count/(self.episode_count+1):.2%}"
        text_surface = font.render(stats_text, True, self.BLACK)
        self.screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()

def main():
    env = MultiAgentGridWorld(num_agents=10)  # Increase/decrease number of agents here
    clock = pygame.time.Clock()
    episodes = 1000
    max_steps = 100
    
    for episode in range(episodes):
        env.episode_count = episode
        states = env.reset()
        all_done = False
        
        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Update all agents
            for i in range(env.num_agents):
                if not env.agents[i].done:
                    action = env.get_action(states[i])
                    next_state, reward, done = env.step(i, action)
                    env.update_q_table(states[i], action, reward, next_state)
                    states[i] = next_state
            
            # Check if all agents are done
            all_done = all(agent.done for agent in env.agents)
            
            env.draw()
            clock.tick(300)  # Increased speed for multiple agents
            
            if all_done:
                break
        
        # Decrease exploration rate
        env.epsilon = max(0.01, env.epsilon * 0.995)
        
        if episode % 10 == 0:
            avg_reward = sum(agent.total_reward for agent in env.agents) / env.num_agents
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Success Rate: {env.success_count/(episode+1):.2%}")

if __name__ == "__main__":
    main()