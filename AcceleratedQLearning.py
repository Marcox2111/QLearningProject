import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from typing import Tuple, Dict, Any, List
from DroneEnvironment import DroneEnvironment

class OptimizedQLearning:
    def __init__(
        self,
        env,
        n_episodes: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.97,
        initial_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
    ):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        
        # Preallocate arrays with numpy for faster access
        self.episode_rewards = np.zeros(n_episodes, dtype=np.float32)
        self.episode_lengths = np.zeros(n_episodes, dtype=np.int32)
        self.epsilon_history = np.zeros(n_episodes, dtype=np.float32)
        
        # Optimize epsilon decay calculation
        self.decay_rate = np.float32(np.exp(np.log(min_epsilon / initial_epsilon) / n_episodes))
        self.epsilon = initial_epsilon
        
        # Use numpy array for Q-values instead of defaultdict
        self.state_map = {}  # Map state tuples to indices
        self.reverse_state_map = []  # Map indices back to state tuples
        self.q_values = None  # Will be initialized when first state is seen
        
        # Pre-calculate action deltas for faster state updates
        self.action_deltas = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int8)

    def _get_state_index(self, state: tuple) -> int:
        """Get or create index for state"""
        if state not in self.state_map:
            if self.q_values is None:
                # Initialize Q-values array with first state
                self.q_values = np.zeros((1000, self.env.num_actions), dtype=np.float32)
            elif len(self.state_map) >= len(self.q_values):
                # Double array size if needed
                self.q_values = np.vstack([
                    self.q_values,
                    np.zeros_like(self.q_values)
                ])
            
            idx = len(self.state_map)
            self.state_map[state] = idx
            self.reverse_state_map.append(state)
            return idx
        return self.state_map[state]

    def choose_action(self, state: tuple) -> int:
        """Vectorized epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.num_actions)
        state_idx = self._get_state_index(state)
        return int(np.argmax(self.q_values[state_idx]))

    def update_q_value(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        """Vectorized Q-value update"""
        state_idx = self._get_state_index(state)
        next_state_idx = self._get_state_index(next_state)
        
        # Vectorized max operation
        best_next_value = np.max(self.q_values[next_state_idx])
        
        # Vectorized Q-value update
        td_target = reward + self.gamma * best_next_value
        self.q_values[state_idx, action] += self.lr * (td_target - self.q_values[state_idx, action])

    def _batch_update(self, states: List[tuple], actions: List[int], 
                     rewards: List[float], next_states: List[tuple]) -> None:
        """Vectorized batch update of Q-values"""
        state_indices = np.array([self._get_state_index(s) for s in states])
        next_state_indices = np.array([self._get_state_index(s) for s in next_states])
        
        # Compute best next values for all transitions at once
        best_next_values = np.max(self.q_values[next_state_indices], axis=1)
        
        # Compute TD targets for all transitions
        td_targets = rewards + self.gamma * best_next_values
        
        # Update Q-values for all transitions
        self.q_values[state_indices, actions] += self.lr * (
            td_targets - self.q_values[state_indices, actions]
        )

    def train(
        self,
        max_episodes: int,
        min_improvement: float = 0.01,
        convergence_window: int = 100,
        patience: int = 10,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Optimized training loop with batch updates"""
        print(f"Starting training for {max_episodes} episodes...")
        
        best_avg = float('-inf')
        best_episode = 0
        last_avg = float('-inf')
        no_improvement_count = 0
        
        # Initialize batch storage
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        
        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Store transition in batch
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_next_states.append(next_state)
                
                # Perform batch update when buffer is full
                if len(batch_states) >= batch_size:
                    self._batch_update(
                        batch_states,
                        batch_actions,
                        batch_rewards,
                        batch_next_states
                    )
                    # Clear batches
                    batch_states = []
                    batch_actions = []
                    batch_rewards = []
                    batch_next_states = []
                
                episode_reward += reward
                steps += 1
                
                if done:
                    break
                state = next_state
            
            # Update metrics
            self.episode_rewards[episode] = episode_reward
            self.episode_lengths[episode] = steps
            self.epsilon_history[episode] = self.epsilon
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            
            # Check convergence every window episodes
            if (episode + 1) % convergence_window == 0:
                current_avg = np.mean(self.episode_rewards[episode-convergence_window+1:episode+1])
                improvement = current_avg - last_avg
                
                if current_avg > best_avg:
                    best_avg = current_avg
                    best_episode = episode
                
                if improvement < min_improvement:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        print("\nTraining converged!")
                        break
                else:
                    no_improvement_count = 0
                
                last_avg = current_avg
                # Print progress less frequently
                self._print_progress(episode, convergence_window, current_avg, 
                                      improvement, best_avg, best_episode)
        
        return self._generate_training_summary(episode, convergence_window, best_avg, best_episode)

    def _print_progress(self, episode: int, window: int, avg: float, 
                       improvement: float, best_avg: float, best_episode: int) -> None:
        print(
            f"Episode: {episode + 1}\n"
            f"Average Reward (last {window}): {avg:.2f}\n"
            f"Improvement: {improvement:.3f}\n"
            f"Epsilon: {self.epsilon:.3f}\n"
            f"Best Average: {best_avg:.2f} (Episode {best_episode + 1})\n"
            f"--------------------"
        )

    def _generate_training_summary(self, episode: int, window: int, 
                                 best_avg: float, best_episode: int) -> Dict[str, Any]:
        return {
            'total_episodes': episode + 1,
            'final_epsilon': self.epsilon,
            'best_average': best_avg,
            'best_episode': best_episode + 1,
            'final_average': np.mean(self.episode_rewards[max(0, episode-window+1):episode+1])
        }

    def plot_training_results(self) -> None:
        valid_episodes = np.where(self.episode_rewards != 0)[0][-1] + 1
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        window = min(100, valid_episodes // 10)
        
        reward_avg = np.convolve(self.episode_rewards[:valid_episodes], 
                               np.ones(window)/window, 'valid')
        length_avg = np.convolve(self.episode_lengths[:valid_episodes], 
                               np.ones(window)/window, 'valid')
        
        ax1.plot(self.episode_rewards[:valid_episodes], alpha=0.3, color='blue', label='Raw')
        ax1.plot(range(window-1, valid_episodes), reward_avg, color='red', 
                label=f'{window}-ep Moving Avg')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        
        ax2.plot(self.episode_lengths[:valid_episodes], alpha=0.3, color='blue', label='Raw')
        ax2.plot(range(window-1, valid_episodes), length_avg, color='red', 
                label=f'{window}-ep Moving Avg')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        
        ax3.plot(self.epsilon_history[:valid_episodes], color='green')
        ax3.set_title('Epsilon Decay')
        
        if self.q_values is not None:
            q_vals = self.q_values[:len(self.state_map)]
            ax4.hist(np.max(q_vals, axis=1), bins=30, color='blue', alpha=0.7)
            ax4.set_title('State Values Distribution')
        
        plt.tight_layout()
        plt.show()

    def save(self, filepath: str) -> None:
        save_data = {
            'q_values': self.q_values[:len(self.state_map)],
            'state_map': self.state_map,
            'reverse_state_map': self.reverse_state_map,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'epsilon': self.epsilon,
            'params': {
                'lr': self.lr,
                'gamma': self.gamma,
                'initial_epsilon': self.initial_epsilon,
                'min_epsilon': self.min_epsilon,
                'decay_rate': self.decay_rate
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self, filepath: str) -> bool:
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.q_values = data['q_values']
            self.state_map = data['state_map']
            self.reverse_state_map = data['reverse_state_map']
            self.episode_rewards = data['episode_rewards']
            self.episode_lengths = data['episode_lengths']
            self.epsilon_history = data['epsilon_history']
            self.epsilon = data['epsilon']
            
            params = data['params']
            self.lr = params['lr']
            self.gamma = params['gamma']
            self.initial_epsilon = params['initial_epsilon']
            self.min_epsilon = params['min_epsilon']
            self.decay_rate = params['decay_rate']
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def run_episode(self, render: bool = True) -> float:
        state = self.env.reset()
        total_reward = 0
        
        while True:
            action = np.argmax(self.q_values[self._get_state_index(state)])
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            
            if done:
                break
                
            state = next_state
        
        return total_reward

def main():
    from DroneEnvironment import DroneEnvironment
    # Create environment and agent
    env = DroneEnvironment(
        grid_size=10,
        max_steps=100,
        safe_return_threshold=0.3
    )

    agent = OptimizedQLearning(
        env=env,
        n_episodes=100000,
        learning_rate=0.1,
        initial_epsilon=1.0,
        min_epsilon=0.01
    )

    training_summary = agent.train(
        max_episodes=100000,
        min_improvement=0.01,
        convergence_window=1000,
        patience=20,
        batch_size=32
    )
    
    agent.save('trained_agent.pkl')
    agent.plot_training_results()
    
if __name__ == "__main__":
    main()