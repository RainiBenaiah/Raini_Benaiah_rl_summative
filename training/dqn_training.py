import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
import gymnasium as gym
from custom_env import BeehiveManagementEnv
from rendering import BeehiveRenderer
import os
import imageio
import matplotlib.pyplot as plt

class DQNTraining:
    """Class to handle DQN training, evaluation, and visualization"""
    
    def __init__(self, num_hives=4, save_path="models/dqn/dqn_model"):
        self.save_path = save_path
        self.env = Monitor(BeehiveManagementEnv(num_hives=num_hives, render_mode="human"))
        self.eval_env = Monitor(BeehiveManagementEnv(num_hives=num_hives, render_mode="human"))
        self.renderer = BeehiveRenderer(width=1200, height=800)
        self.episode_rewards = []
        self.training_losses = []
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("gifs", exist_ok=True)

    def train(self):
        """Train DQN model"""
        hyperparams = {
            "learning_rate": 5e-4,  # Reduced for stability
            "buffer_size": 100000,
            "batch_size": 128,  # Increased for stable gradients
            "gamma": 0.99,
            "exploration_fraction": 0.2,  # Increased for better exploration
            "exploration_final_eps": 0.05,  # Increased to avoid premature convergence
            "learning_starts": 1000,
            "target_update_interval": 1000,
            "train_freq": 8,  # Adjusted for balance
        }

        model = DQN(
            policy="MlpPolicy",
            env=self.env,
            **hyperparams,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )

        class LossCallback(EvalCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.training_losses = []

            def _on_step(self) -> bool:
                # Access loss from self.locals['infos']
                if self.locals.get('infos') and isinstance(self.locals['infos'], list):
                    for info in self.locals['infos']:
                        if 'loss' in info:
                            self.training_losses.append(info['loss'])
                return super()._on_step()

        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
        eval_callback = LossCallback(
            self.eval_env,
            best_model_save_path=os.path.dirname(self.save_path),
            log_path="./eval_logs/",
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
            callback_after_eval=stop_callback
        )

        model.learn(
            total_timesteps=200000,  # Increased for more training
            callback=eval_callback,
            progress_bar=True
        )

        model.save(self.save_path)
        self.training_losses = eval_callback.training_losses
        print(f"DQN model saved to {self.save_path}")
        if not self.training_losses:
            print("Warning: No training losses logged during training")

        return model

    def evaluate(self, model, n_episodes=5):
        """Evaluate trained DQN model"""
        episode_rewards = []
        episode_lengths = []

        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            self.episode_rewards.append(total_reward)

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "episode_rewards": episode_rewards
        }

    def record_gif(self, model, episodes=3, max_steps=1000):
        """Record GIF for specified number of episodes"""
        for ep in range(episodes):
            frames = []
            obs, _ = self.eval_env.reset()
            done = False
            steps = 0
            total_reward = 0

            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                total_reward += reward
                frame = self.renderer.render(self.eval_env.unwrapped, action, reward)
                frames.append(np.transpose(frame, (1, 0, 2)))
                steps += 1
                done = terminated or truncated

            filename = f"gifs/dqn_episode_{ep+1}.gif"
            imageio.mimsave(filename, [frame.astype(np.uint8) for frame in frames], fps=8)
            print(f"DQN GIF for episode {ep+1} saved as {filename}, Total Reward: {total_reward:.2f}")

    def plot_results(self):
        """Plot cumulative rewards and training stability"""
        plt.figure(figsize=(10, 5))
        if self.episode_rewards:
            plt.plot(np.cumsum(self.episode_rewards), label="DQN Cumulative Reward")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title("DQN Cumulative Reward Over Episodes")
            plt.legend()
            plt.savefig("plots/dqn_cumulative_reward.png")
            plt.close()
        else:
            print("Warning: No episode rewards available for cumulative reward plot")

        plt.figure(figsize=(10, 5))
        if self.training_losses:
            plt.plot(self.training_losses, label="DQN Loss")
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("DQN Training Stability")
            plt.legend()
            plt.savefig("plots/dqn_training_stability.png")
            plt.close()
        else:
            print("Warning: No training losses available for stability plot")

if __name__ == "__main__":
    trainer = DQNTraining()
    model = trainer.train()
    results = trainer.evaluate(model)
    trainer.record_gif(model, episodes=3)
    trainer.plot_results()
    
    print("\nDQN Evaluation Results:")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.2f}")