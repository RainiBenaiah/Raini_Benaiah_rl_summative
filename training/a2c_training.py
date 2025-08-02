import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from custom_env import BeehiveManagementEnv
from rendering import BeehiveRenderer
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygame
import torch
from torch.distributions.categorical import Categorical

class A2CTraining:
    """Class to handle A2C training, evaluation, and visualization"""
    
    def __init__(self, num_hives=4, save_path="models/pg/ac_model.zip"):
        self.save_path = save_path
        self.env = Monitor(BeehiveManagementEnv(num_hives=num_hives, render_mode="human"))
        self.eval_env = Monitor(BeehiveManagementEnv(num_hives=num_hives, render_mode="human"))
        self.renderer = BeehiveRenderer(width=1200, height=800)
        self.episode_rewards = []
        self.policy_entropies = []
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("gifs", exist_ok=True)

    def train(self):
        """Train A2C model"""
        hyperparams = {
            "learning_rate": 5e-4,
            "n_steps": 10,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "vf_coef": 0.5,
            "ent_coef": 0.03,  # Increased to encourage exploration
        }

        model = A2C(
            policy="MlpPolicy",
            env=self.env,
            **hyperparams,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )

        class EntropyCallback(EvalCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.entropies = []
                self.model = None

            def _on_training_start(self):
                self.model = self.locals['self']  # Access the A2C model

            def _on_step(self) -> bool:
                if self.model and hasattr(self.model, 'policy'):
                    try:
                        obs = self.locals.get('new_obs')
                        if obs is not None:
                            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.model.device)
                            with torch.no_grad():
                                dist = self.model.policy.get_distribution(obs_tensor)
                                entropy = dist.distribution.entropy().mean().item()
                                self.entropies.append(entropy)
                                print(f"Logged entropy: {entropy:.4f}")
                    except Exception as e:
                        print(f"Entropy logging error: {e}")
                return super()._on_step()

        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
        eval_callback = EntropyCallback(
            self.eval_env,
            best_model_save_path=os.path.dirname(self.save_path),
            log_path="./eval_logs/",
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
            callback_after_eval=stop_callback
        )

        model.learn(
            total_timesteps=500000,  # Increased for better learning
            callback=eval_callback,
            progress_bar=True
        )

        model.save(self.save_path)
        self.policy_entropies = eval_callback.entropies
        print(f"A2C model saved to {self.save_path}")
        if not self.policy_entropies:
            print("Warning: No policy entropies logged during training")

        return model

    def evaluate(self, model, n_episodes=5):
        """Evaluate trained A2C model"""
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
            self.episode_rewards.append(total_reward)
            episode_lengths.append(steps)

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

            filename = f"gifs/ac_episode_{ep+1}.gif"
            imageio.mimsave(filename, [frame.astype(np.uint8) for frame in frames], fps=8)
            print(f"A2C GIF for episode {ep+1} saved as {filename}, Total Reward: {total_reward:.2f}")

    def close(self):
        """Clean up renderer resources"""
        if hasattr(self, 'renderer'):
            self.renderer.close()

    def plot_results(self):
        """Plot cumulative rewards and training stability"""
        plt.figure(figsize=(10, 5))
        if self.episode_rewards:
            plt.plot(np.cumsum(self.episode_rewards), label="A2C Cumulative Reward")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title("A2C Cumulative Reward Over Episodes")
            plt.legend()
            plt.savefig("plots/ac_cumulative_reward.png")
            plt.close()
        else:
            print("Warning: No episode rewards available for cumulative reward plot")

        plt.figure(figsize=(10, 5))
        if self.policy_entropies:
            plt.plot(self.policy_entropies, label="A2C Policy Entropy")
            plt.xlabel("Training Step")
            plt.ylabel("Entropy")
            plt.title("A2C Training Stability")
            plt.legend()
            plt.savefig("plots/ac_training_stability.png")
            plt.close()
        else:
            print("Warning: No policy entropies available for stability plot")

if __name__ == "__main__":
    try:
        pygame.init()
        trainer = A2CTraining()
        model = trainer.train()
        results = trainer.evaluate(model)
        trainer.record_gif(model, episodes=3)
        trainer.plot_results()
        
        print("\nA2C Evaluation Results:")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Episode Length: {results['mean_length']:.2f}")
    finally:
        pygame.quit()