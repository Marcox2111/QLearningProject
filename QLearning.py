import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from DroneEnvironment import DroneEnvironment
import pickle
import json

class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table as a defaultdict
        self.q_table = defaultdict(lambda: np.zeros(env.num_actions))
        
        # For tracking learning progress
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []


    def save_qtable(self, filepath):
        """Save Q-table to a file"""
        # Convert defaultdict to regular dict and convert numpy arrays to lists
        q_dict = {str(state): list(actions) for state, actions in self.q_table.items()}
        
        # Save using pickle (preserves tuple keys)
        with open(filepath, 'wb') as f:
            pickle.dump(q_dict, f)
        
        print(f"Q-table saved to {filepath}")

    def load_qtable(self, filepath):
        """Load Q-table from a file"""
        try:
            # Load the dictionary using pickle
            with open(filepath, 'rb') as f:
                q_dict = pickle.load(f)
            
            # Convert back to defaultdict with numpy arrays
            self.q_table = defaultdict(lambda: np.zeros(self.env.num_actions))
            for state_str, actions in q_dict.items():
                # Evaluate the string representation of the state tuple
                state = eval(state_str)
                self.q_table[state] = np.array(actions)
            
            print(f"Q-table loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            return False

    def save_training_progress(self, filepath):
        """Save training progress (rewards and episode lengths)"""
        progress = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon': self.epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(progress, f)
        print(f"Training progress saved to {filepath}")

    def load_training_progress(self, filepath):
        """Load training progress"""
        try:
            with open(filepath, 'rb') as f:
                progress = pickle.load(f)
            self.episode_rewards = progress['episode_rewards']
            self.episode_lengths = progress['episode_lengths']
            self.epsilon = progress['epsilon']
            print(f"Training progress loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading training progress: {e}")
            return False
    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.num_actions)
        else:
            return np.argmax(self.q_table[state])

    def train(self, max_episodes=100000, min_improvement=0.01, convergence_window=500, patience=10):
        """
        Train the agent with convergence criteria
        
        Args:
            max_episodes: Maximum number of episodes to train for
            min_improvement: Minimum improvement in average reward to continue training
            convergence_window: Number of episodes to average over for convergence check
            patience: Number of checks without improvement before stopping
        """
        print(f"Starting training (max episodes: {max_episodes})...")
        print(f"Convergence criteria: {min_improvement:.3f} min improvement over {convergence_window} episodes")
        
        last_avg = float('-inf')
        consecutive_no_improvement = 0
        best_avg = float('-inf')
        best_episode = 0
        
        for episode in range(max_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            done = False
            while not done:
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Update Q-value
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.lr * td_error
                
                # Move to next state
                state = next_state
                total_reward += reward
                steps += 1
            
            self.epsilon_history.append(self.epsilon)
            # Decay epsilon
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon - (self.epsilon - self.min_epsilon) / max_episodes
            )
            
            # Track progress
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Check convergence every convergence_window episodes
            if (episode + 1) % convergence_window == 0:
                current_avg = np.mean(self.episode_rewards[-convergence_window:])
                improvement = current_avg - last_avg
                
                # Keep track of best performance
                if current_avg > best_avg:
                    best_avg = current_avg
                    best_episode = episode
                
                # Print progress
                print(f"Episode: {episode + 1}")
                print(f"Average Reward (last {convergence_window}): {current_avg:.2f}")
                print(f"Improvement: {improvement:.3f}")
                print(f"Average Steps (last {convergence_window}): {np.mean(self.episode_lengths[-convergence_window:]):.2f}")
                print(f"Epsilon: {self.epsilon:.3f}")
                print(f"Best Average so far: {best_avg:.2f} (Episode {best_episode + 1})")
                print("--------------------")
                
                # Check if improvement is below threshold
                if improvement < min_improvement:
                    consecutive_no_improvement += 1
                    print(f"No significant improvement for {consecutive_no_improvement} checks")
                    
                    if consecutive_no_improvement >= patience:
                        print("\nTraining converged!")
                        print(f"Final average reward: {current_avg:.2f}")
                        print(f"Best average reward: {best_avg:.2f} (Episode {best_episode + 1})")
                        print(f"Total episodes trained: {episode + 1}")
                        return
                else:
                    consecutive_no_improvement = 0
                
                last_avg = current_avg
        
        print("\nReached maximum episodes!")
        print(f"Final average reward: {np.mean(self.episode_rewards[-convergence_window:]):.2f}")
        print(f"Best average reward: {best_avg:.2f} (Episode {best_episode + 1})")

    def plot_training_results(self):
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        
        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        
        plt.tight_layout()
        plt.show()

    def run_episode(self, render=True):
        """Run a single episode using the learned policy"""
        state = self.env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Choose action greedily
            action = np.argmax(self.q_table[state])
            state, reward, done = self.env.step(action)
            total_reward += reward
            
        return total_reward
    
    def plot_training_results(self):
        """Create comprehensive training visualization with multiple metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Moving average of rewards
        window_size = 100
        moving_avg = np.convolve(self.episode_rewards, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        ax1.plot(self.episode_rewards, alpha=0.3, color='blue', label='Raw')
        ax1.plot(range(window_size-1, len(self.episode_rewards)), 
                moving_avg, color='red', label=f'{window_size}-ep Moving Avg')
        ax1.set_title('Episode Rewards Over Time')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Episode lengths
        moving_avg_length = np.convolve(self.episode_lengths,
                                    np.ones(window_size)/window_size,
                                    mode='valid')
        ax2.plot(self.episode_lengths, alpha=0.3, color='blue', label='Raw')
        ax2.plot(range(window_size-1, len(self.episode_lengths)),
                moving_avg_length, color='red', label=f'{window_size}-ep Moving Avg')
        ax2.set_title('Episode Lengths Over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Epsilon decay

        ax3.plot(self.epsilon_history, color='green')
        ax3.set_title('Epsilon Decay Over Time')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True, alpha=0.3)

        # 4. Q-value statistics
        q_values = np.array([list(values) for values in self.q_table.values()])
        if len(q_values) > 0:
            max_q = [np.max(q_values)]
            mean_q = [np.mean(q_values)]
            min_q = [np.min(q_values)]
            
            ax4.plot(max_q, color='green', label='Max Q-value')
            ax4.plot(mean_q, color='blue', label='Mean Q-value')
            ax4.plot(min_q, color='red', label='Min Q-value')
            ax4.set_title('Q-value Statistics')
            ax4.set_xlabel('State-Action Pairs')
            ax4.set_ylabel('Q-value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_training_statistics(self):
        """Calculate and return key training statistics"""
        if not self.episode_rewards:
            return "No training data available"
        
        stats = {
            'total_episodes': len(self.episode_rewards),
            'final_epsilon': self.epsilon,
            'last_100_episodes': {
                'avg_reward': np.mean(self.episode_rewards[-100:]),
                'avg_length': np.mean(self.episode_lengths[-100:]),
                'min_reward': np.min(self.episode_rewards[-100:]),
                'max_reward': np.max(self.episode_rewards[-100:])
            },
            'overall': {
                'avg_reward': np.mean(self.episode_rewards),
                'avg_length': np.mean(self.episode_lengths),
                'min_reward': np.min(self.episode_rewards),
                'max_reward': np.max(self.episode_rewards)
            }
        }
        
        # Q-table statistics
        q_values = np.array([list(values) for values in self.q_table.values()])
        if len(q_values) > 0:
            stats['q_values'] = {
                'max': float(np.max(q_values)),
                'min': float(np.min(q_values)),
                'mean': float(np.mean(q_values)),
                'std': float(np.std(q_values))
            }
        
        return stats

    def plot_action_distribution(self):
        """Plot the distribution of preferred actions across states"""
        if not self.q_table:
            print("No Q-table data available")
            return
        
        # Get preferred actions for each state
        preferred_actions = [np.argmax(actions) for actions in self.q_table.values()]
        
        # Create action distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(preferred_actions, bins=range(self.env.num_actions + 1),
                align='left', rwidth=0.8)
        plt.title('Distribution of Preferred Actions Across States')
        plt.xlabel('Action')
        plt.ylabel('Number of States')
        plt.xticks(range(self.env.num_actions))
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_state_values(self):
        """Plot the distribution of state values"""
        if not self.q_table:
            print("No Q-table data available")
            return
        
        # Calculate state values (max Q-value for each state)
        state_values = [np.max(actions) for actions in self.q_table.values()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(state_values, bins=30)
        plt.title('Distribution of State Values')
        plt.xlabel('State Value (Max Q-value)')
        plt.ylabel('Number of States')
        plt.grid(True, alpha=0.3)
        plt.show()

def main():
    # Create environment and agent
    env = DroneEnvironment(
        grid_size=10,
        max_steps=50,
        # render_mode="human",
        safe_return_threshold=0.3
    )
    
    agent = QLearning(
        env=env,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_decay=0.9995,  # Slower decay
        min_epsilon=0.05      # Higher minimum
    )

    agent.train(
        max_episodes=100000,
        min_improvement=0.01,
        convergence_window=500,  # Larger window
        patience=10             # More patience
    )
    
    # Save Q-table and progress
    agent.save_qtable('drone_qtable.pkl')
    agent.save_training_progress('training_progress.pkl')

    agent.plot_training_results()  # Shows the comprehensive 4-panel plot
    print(agent.get_training_statistics())  # Prints detailed statistics

    # Analyze action and state value distributions
    agent.plot_action_distribution()
    agent.plot_state_values()

    env = DroneEnvironment(
        grid_size=10,
        max_steps=50,
        render_mode="human",
        safe_return_threshold=0.3
    )

    new_agent = QLearning(env)

    
    # Load saved Q-table and progress
    if new_agent.load_qtable('drone_qtable.pkl'):
        if new_agent.load_training_progress('training_progress.pkl'):
            print("\nRunning episodes with loaded Q-table...")
            for i in range(5):
                reward = new_agent.run_episode(render=True)
                print(f"Test Episode {i+1} Reward: {reward:.2f}")
    
    env.close()

# Example of loading and running separately
def run_trained_agent():
    """Function to run a trained agent without training"""
    env = DroneEnvironment(
        grid_size=10,
        max_steps=50,
        render_mode="human",
        safe_return_threshold=0.3
    )
    
    agent = QLearning(env)
    
    # Load the saved Q-table
    if agent.load_qtable('drone_qtable.pkl'):
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