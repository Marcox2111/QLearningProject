import numpy as np
from typing import Tuple, List, Dict
import time

class ValueIteration:
    def __init__(self, env, gamma: float = 0.95):
        self.env = env
        self.gamma = gamma
        
        # Simplified state representation
        self.V = {}  # State-value dictionary
        self.policy = {}  # State-action dictionary
        
        # Pre-compute all possible states
        self.all_states = []
        for x in range(env.size):
            for y in range(env.size):
                for visited_enc in range(2 ** len(env.targets)):
                    self.all_states.append((x, y, visited_enc))
                    self.V[(x, y, visited_enc)] = 0.0
                    self.policy[(x, y, visited_enc)] = 0
    
    def _encode_visited_targets(self, visited_targets: set) -> int:
        """Convert visited targets set to integer encoding."""
        encoding = 0
        targets_list = list(self.env.targets)
        for target in visited_targets:
            if target in targets_list:
                idx = targets_list.index(target)
                encoding |= (1 << idx)
        return encoding
    
    def _decode_visited_targets(self, encoding: int) -> set:
        """Convert integer encoding back to visited targets set."""
        visited = set()
        targets_list = list(self.env.targets)
        for i, target in enumerate(targets_list):
            if encoding & (1 << i):
                visited.add(target)
        return visited

    def train(self, max_iterations: int = 100, threshold: float = 0.01):
        """Train the agent using value iteration."""
        iteration = 0
        while iteration < max_iterations:
            delta = 0
            print(f"Training iteration {iteration}")
            
            # Update each state
            for state in self.all_states:
                x, y, visited_enc = state
                old_value = self.V[state]
                
                # Try all actions
                action_values = []
                
                # Save environment state
                old_pos = self.env.agent_pos
                old_steps = self.env.current_steps
                old_visited = self.env.visited_targets.copy()
                
                # Set current state
                self.env.agent_pos = (x, y)
                self.env.visited_targets = self._decode_visited_targets(visited_enc)
                
                for action in range(4):
                    next_state, reward, done, _ = self.env.step(action)
                    
                    if not done:
                        next_visited_enc = self._encode_visited_targets(self.env.visited_targets)
                        next_state_tuple = (next_state[0], next_state[1], next_visited_enc)
                        value = reward + self.gamma * self.V[next_state_tuple]
                    else:
                        value = reward
                    
                    action_values.append(value)
                    
                    # Reset for next action
                    self.env.agent_pos = (x, y)
                    self.env.visited_targets = self._decode_visited_targets(visited_enc)
                    self.env.current_steps = old_steps
                
                # Update value and policy
                best_value = max(action_values)
                best_action = np.argmax(action_values)
                
                self.V[state] = best_value
                self.policy[state] = best_action
                
                delta = max(delta, abs(old_value - best_value))
                
                # Restore environment state
                self.env.agent_pos = old_pos
                self.env.current_steps = old_steps
                self.env.visited_targets = old_visited
            
            print(f"Delta: {delta}")
            
            if delta < threshold:
                print(f"Converged after {iteration} iterations!")
                break
            
            iteration += 1

    def run_episode(self, render: bool = True) -> Tuple[float, List[Tuple[int, int]]]:
        """Run one episode using the learned policy."""
        state, _ = self.env.reset()
        total_reward = 0
        trajectory = [state]
        
        while True:
            if render:
                self.env.render(total_reward)
                time.sleep(0.2)
            
            # Get state encoding
            visited_enc = self._encode_visited_targets(self.env.visited_targets)
            state_tuple = (state[0], state[1], visited_enc)
            
            # Get action from policy
            action = self.policy[state_tuple]
            
            # Take action
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            trajectory.append(state)
            
            if done:
                if render:
                    self.env.render(total_reward)
                    time.sleep(1)
                break
        
        return total_reward, trajectory

if __name__ == "__main__":
    from DroneEnvironment import GridEnvironment
    
    # Create environment and agent
    env = GridEnvironment(size=10, max_steps=50)
    agent = ValueIteration(env)
    
    # Train the agent
    print("Training agent...")
    agent.train(max_iterations=50, threshold=0.1)
    
    # Test the learned policy
    print("\nTesting learned policy...")
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        reward, trajectory = agent.run_episode(render=True)
        print(f"Episode finished with reward: {reward:.2f}")
        print(f"Trajectory length: {len(trajectory)}")
        print(f"Visited {len(env.visited_targets)}/{len(env.targets)} targets")
    
    env.close()