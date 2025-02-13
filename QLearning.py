import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from typing import Tuple, Dict, Any
from DroneEnvironment import DroneEnvironment

class QLearning:
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
        
        # Preallocate arrays for storing metrics
        self.episode_rewards = np.zeros(n_episodes)
        self.episode_lengths = np.zeros(n_episodes)
        self.epsilon_history = np.zeros(n_episodes)
        
        self.warmup_episodes = 30000
        effective_episodes = max(1, n_episodes - self.warmup_episodes)
        self.decay_rate = np.exp(np.log(min_epsilon / initial_epsilon) / (effective_episodes/2))
        self.epsilon = initial_epsilon
        
        # Initialize Q-table with optimized defaultdict
        self._q_values = defaultdict(lambda: np.zeros(env.num_actions, dtype=np.float32))
        
        
        # Cache for statistics
        self._stats_cache = {}
        self._last_updated_episode = -1
        
    def softmax_selection(self, state: tuple, temperature=1.0) -> int:
        """Softmax (Boltzmann) exploration strategy."""
        q_values = self._q_values[state]
        # Use a stable softmax approach to avoid numerical overflow
        q_adjusted = (q_values - np.max(q_values)) / max(temperature, 1e-5)
        preferences = np.exp(q_adjusted)
        probs = preferences / np.sum(preferences)
        return np.random.choice(range(self.env.num_actions), p=probs)
    
    def decay_epsilon(self, current_episode: int) -> None:
        """Epsilon decay with warmup period"""
        if current_episode < self.warmup_episodes:
            # During warmup, maintain initial epsilon
            self.epsilon = self.initial_epsilon
        else:
            # After warmup, apply decay from initial value
            decay_episode = current_episode - self.warmup_episodes
            self.epsilon = max(self.min_epsilon, 
                             self.initial_epsilon * self.decay_rate ** decay_episode)

    def choose_action(self, state: tuple) -> int:
        """Epsilon-greedy action selection with numpy optimizations"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.num_actions)
        return int(np.argmax(self._q_values[state]))

    def update_q_value(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        """Optimized Q-value update"""
        next_state_values = self._q_values[next_state]
        best_next_value = next_state_values.max()
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - self._q_values[state][action]
        self._q_values[state][action] += self.lr * td_error

    def train(
        self,
        max_episodes: int,
        min_improvement: float = 0.01,
        convergence_window: int = 100,
        patience: int = 10
    ) -> Dict[str, Any]:
        """Optimized training loop with early stopping"""
        print(f"Starting training for {max_episodes} episodes...")
        
        best_avg = float('-inf')
        best_episode = 0
        last_avg = float('-inf')
        no_improvement_count = 0
        window_rewards = np.zeros(convergence_window)
        temperature = 0.5
        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            # Main episode loop
            while True:
                action = self.choose_action(state)
                # action = self.softmax_selection(state, temperature=temperature)
                next_state, reward, done = self.env.step(action)
                
                self.update_q_value(state, action, reward, next_state)
                
                episode_reward += reward
                steps += 1
                
                if done:
                    break
                    
                state = next_state
            
            # Update metrics (using pre-allocated arrays)
            self.episode_rewards[episode] = episode_reward
            self.episode_lengths[episode] = steps
            self.epsilon_history[episode] = self.epsilon
            
            # Decay epsilon
            self.decay_epsilon(episode)
            
            # Convergence check
            if (episode + 1) % convergence_window == 0:
                current_window = self.episode_rewards[episode-convergence_window+1:episode+1]
                current_avg = current_window.mean()
                improvement = current_avg - last_avg
                
                if current_avg > best_avg:
                    best_avg = current_avg
                    best_episode = episode
                
                # Print progress efficiently
                self._print_progress(episode, convergence_window, current_avg, 
                                  improvement, best_avg, best_episode)
                
                if improvement < min_improvement:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        print("\nTraining converged!")
                        break
                else:
                    no_improvement_count = 0
                
                last_avg = current_avg
        
        return self._generate_training_summary(episode, convergence_window, best_avg, best_episode)

    def _print_progress(self, episode: int, window: int, avg: float, 
                       improvement: float, best_avg: float, best_episode: int) -> None:
        """Efficient progress printing"""
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
        """Generate efficient training summary"""
        return {
            'total_episodes': episode + 1,
            'final_epsilon': self.epsilon,
            'best_average': best_avg,
            'best_episode': best_episode + 1,
            'final_average': np.mean(self.episode_rewards[max(0, episode-window+1):episode+1])
        }

    def plot_training_results(self) -> None:
        """Optimized plotting with efficient data processing"""
        valid_episodes = np.where(self.episode_rewards != 0)[0][-1] + 1
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        window = min(100, valid_episodes // 10)
        
        # Efficient moving averages
        reward_avg = np.convolve(self.episode_rewards[:valid_episodes], 
                               np.ones(window)/window, 'valid')
        length_avg = np.convolve(self.episode_lengths[:valid_episodes], 
                               np.ones(window)/window, 'valid')
        
        # Plot rewards
        ax1.plot(self.episode_rewards[:valid_episodes], alpha=0.3, color='blue', label='Raw')
        ax1.plot(range(window-1, valid_episodes), reward_avg, color='red', 
                label=f'{window}-ep Moving Avg')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        
        # Plot lengths
        ax2.plot(self.episode_lengths[:valid_episodes], alpha=0.3, color='blue', label='Raw')
        ax2.plot(range(window-1, valid_episodes), length_avg, color='red', 
                label=f'{window}-ep Moving Avg')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        
        # Plot epsilon
        ax3.plot(self.epsilon_history[:valid_episodes], color='green')
        ax3.set_title('Epsilon Decay')
        
        # Plot Q-value statistics if available
        if self._q_values:
            q_array = np.array([list(values) for values in self._q_values.values()])
            if len(q_array) > 0:
                ax4.hist(q_array.max(axis=1), bins=30, color='blue', alpha=0.7)
                ax4.set_title('State Values Distribution')
        
        plt.tight_layout()
        plt.show()

    def save(self, filepath: str) -> None:
        """Efficient single-file saving of all data"""
        save_data = {
            'q_values': dict(self._q_values),
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
        """Efficient single-file loading of all data"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Restore Q-values
            self._q_values.clear()
            self._q_values.update(data['q_values'])
            
            # Restore arrays and parameters
            self.episode_rewards = data['episode_rewards']
            self.episode_lengths = data['episode_lengths']
            self.epsilon_history = data['epsilon_history']
            self.epsilon = data['epsilon']
            
            # Restore parameters
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
        """Run a single episode using the learned policy"""
        state = self.env.reset()
        total_reward = 0
        
        while True:
            action = np.argmax(self._q_values[state])
            state, reward, done = self.env.step(action)
            total_reward += reward
            if done:
                break
            
        return total_reward

def main():
    # Create environment and agent
    env = DroneEnvironment(
        grid_size=10,
        max_steps=100,
        safe_return_threshold=0.3
    )
    
    agent = QLearning(
        env=env,
        n_episodes=500000,
        learning_rate=0.1,
        discount_factor=0.95,
        initial_epsilon=0.2,
        min_epsilon=0.05
    )

    # Train the agent
    training_summary = agent.train(
        max_episodes=500000,
        min_improvement=0.01,
        convergence_window=1000,
        patience=20
    )
    
    # Save all data
    agent.save('trained_agent.pkl')
    print("Training summary:", training_summary)

    # Plot results
    agent.plot_training_results()

    # Create environment for visualization
    viz_env = DroneEnvironment(
        grid_size=10,
        max_steps=100,
        render_mode="human",
        safe_return_threshold=0.3
    )

    # Create new agent for visualization
    viz_agent = QLearning(
        env=viz_env,
        n_episodes=100000,  # This won't be used since we're loading trained data
        learning_rate=0.15,
        initial_epsilon=0.01,  # Start with minimal exploration for visualization
        min_epsilon=0.05
    )
    
    # Load and run trained agent
    if viz_agent.load('trained_agent.pkl'):
        print("\nRunning episodes with trained agent...")
        for i in range(5):
            reward = viz_agent.run_episode(render=True)
            print(f"Test Episode {i+1} Reward: {reward:.2f}")
    
    viz_env.close()

def run_trained_agent():
    """Function to run a trained agent without training"""
    env = DroneEnvironment(
        grid_size=10,
        max_steps=100,
        render_mode="human",
        safe_return_threshold=0.3
    )
    
    agent = QLearning(
        env=env,
        n_episodes=10000,  # This won't be used since we're loading trained data
        learning_rate=0.1,
        initial_epsilon=0.01,  # Start with minimal exploration
        min_epsilon=0.09
    )
    
    # Load and run the trained agent
    if agent.load('trained_agent.pkl'):
        print("Running trained agent...")
        for i in range(5):
            reward = agent.run_episode(render=True)
            print(f"Episode {i+1} Reward: {reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # Choose whether to train or run trained agent
    train_new = input("Train new agent? (y/n): ").lower() == 'y'
    
    if train_new:
        main()
    else:
        run_trained_agent()