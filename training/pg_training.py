import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from custom_env import BeehiveManagementEnv
from rendering import BeehiveRenderer
import imageio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend to avoid Tcl/Tk issues
import matplotlib.pyplot as plt
import pygame

class PolicyNetwork(nn.Module):
    """Simple policy network for REINFORCE"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class REINFORCE:
    """Custom REINFORCE implementation"""
    def __init__(self, env, learning_rate=5e-4, gamma=0.95):
        self.env = env
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.episode_rewards = []
        self.policy_entropies = []
        self.renderer = BeehiveRenderer(width=1200, height=800)

    def train(self, total_timesteps=500000):
        obs, _ = self.env.reset()
        episode_rewards = []
        log_probs = []
        rewards = []
        step = 0

        while step < total_timesteps:
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            action_probs = self.policy(obs_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            obs = next_obs
            episode_rewards.append(reward)
            step += 1

            if terminated or truncated:
                self.episode_rewards.append(sum(episode_rewards))
                returns = []
                R = 0
                for r in rewards[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)
                returns = torch.FloatTensor(returns).to(self.device)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                policy_loss = []
                for log_prob, R in zip(log_probs, returns):
                    policy_loss.append(-log_prob * R)
                policy_loss = torch.stack(policy_loss).sum()

                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()

                entropy = dist.entropy().mean().item()
                self.policy_entropies.append(entropy)

                obs, _ = self.env.reset()
                episode_rewards = []
                log_probs = []
                rewards = []

        os.makedirs("models/pg", exist_ok=True)
        torch.save(self.policy.state_dict(), "models/pg/reinforce_model.pth")
        print("REINFORCE model saved to models/pg/reinforce_model.pth")

    def evaluate(self, model=None, n_episodes=5):
        """Evaluate REINFORCE model using deterministic actions"""
        if model is None:
            model = self
        episode_rewards = []
        episode_lengths = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                action_probs = model.policy(obs_tensor)
                action = torch.argmax(action_probs).item()  # Use deterministic action
                obs, reward, terminated, truncated, _ = self.env.step(action)
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

    def record_gif(self, model=None, episodes=3, max_steps=1000):
        if model is None:
            model = self
        for ep in range(episodes):
            frames = []
            obs, _ = self.env.reset()
            done = False
            steps = 0
            total_reward = 0

            while not done and steps < max_steps:
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                action_probs = model.policy(obs_tensor)
                action = torch.argmax(action_probs).item()  # Use deterministic action
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                frame = self.renderer.render(self.env.unwrapped, action, reward)
                frames.append(np.transpose(frame, (1, 0, 2)))
                steps += 1
                done = terminated or truncated

            filename = f"gifs/reinforce_episode_{ep+1}.gif"
            imageio.mimsave(filename, [frame.astype(np.uint8) for frame in frames], fps=8)
            print(f"REINFORCE GIF for episode {ep+1} saved as {filename}, Total Reward: {total_reward:.2f}")

    def close(self):
        """Clean up renderer resources"""
        if hasattr(self, 'renderer'):
            self.renderer.close()

    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.episode_rewards), label="REINFORCE Cumulative Reward")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title("REINFORCE Cumulative Reward Over Episodes")
        plt.legend()
        plt.savefig("plots/reinforce_cumulative_reward.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.policy_entropies, label="REINFORCE Policy Entropy")
        plt.xlabel("Training Step")
        plt.ylabel("Entropy")
        plt.title("REINFORCE Training Stability")
        plt.legend()
        plt.savefig("plots/reinforce_training_stability.png")
        plt.close()

class PPOTraining:
    def __init__(self, num_hives=4, save_path="models/pg/ppo_model"):
        self.save_path = save_path
        self.env = Monitor(BeehiveManagementEnv(num_hives=num_hives, render_mode="human"))
        self.eval_env = Monitor(BeehiveManagementEnv(num_hives=num_hives, render_mode="human"))
        self.renderer = BeehiveRenderer(width=1200, height=800)
        self.episode_rewards = []
        self.policy_entropies = []

    def train(self):
        hyperparams = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        }

        model = PPO(
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

            def _on_step(self) -> bool:
                if hasattr(self.model, 'policy'):
                    try:
                        dist = self.model.policy.get_distribution(self.model.rollout_buffer.observations)
                        entropy = dist.entropy().mean().item()
                        self.entropies.append(entropy)
                    except Exception:
                        pass  # Skip if distribution is unavailable
                return super()._on_step()

        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
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
            total_timesteps=100000,
            callback=eval_callback,
            progress_bar=True
        )

        model.save(self.save_path)
        self.policy_entropies = eval_callback.entropies
        print(f"PPO model saved to {self.save_path}")

        return model

    def evaluate(self, model, n_episodes=5):
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

            filename = f"gifs/ppo_episode_{ep+1}.gif"
            imageio.mimsave(filename, [frame.astype(np.uint8) for frame in frames], fps=8)
            print(f"PPO GIF for episode {ep+1} saved as {filename}, Total Reward: {total_reward:.2f}")

    def close(self):
        """Clean up renderer resources"""
        if hasattr(self, 'renderer'):
            self.renderer.close()

    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.episode_rewards), label="PPO Cumulative Reward")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title("PPO Cumulative Reward Over Episodes")
        plt.legend()
        plt.savefig("plots/ppo_cumulative_reward.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.policy_entropies, label="PPO Policy Entropy")
        plt.xlabel("Training Step")
        plt.ylabel("Entropy")
        plt.title("PPO Training Stability")
        plt.legend()
        plt.savefig("plots/ppo_training_stability.png")
        plt.close()

class A2CTraining:
    def __init__(self, num_hives=4, save_path="models/pg/a2c_model"):
        self.save_path = save_path
        self.env = Monitor(BeehiveManagementEnv(num_hives=num_hives, render_mode="human"))
        self.eval_env = Monitor(BeehiveManagementEnv(num_hives=num_hives, render_mode="human"))
        self.renderer = BeehiveRenderer(width=1200, height=800)
        self.episode_rewards = []
        self.policy_entropies = []

    def train(self):
        hyperparams = {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
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

            def _on_step(self) -> bool:
                if hasattr(self.model, 'policy'):
                    try:
                        dist = self.model.policy.get_distribution(self.model.rollout_buffer.observations)
                        entropy = dist.entropy().mean().item()
                        self.entropies.append(entropy)
                    except Exception:
                        pass  # Skip if distribution is unavailable
                return super()._on_step()

        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
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
            total_timesteps=100000,
            callback=eval_callback,
            progress_bar=True
        )

        model.save(self.save_path)
        self.policy_entropies = eval_callback.entropies
        print(f"A2C model saved to {self.save_path}")

        return model

    def evaluate(self, model, n_episodes=5):
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

            filename = f"gifs/a2c_episode_{ep+1}.gif"
            imageio.mimsave(filename, [frame.astype(np.uint8) for frame in frames], fps=8)
            print(f"A2C GIF for episode {ep+1} saved as {filename}, Total Reward: {total_reward:.2f}")

    def close(self):
        """Clean up renderer resources"""
        if hasattr(self, 'renderer'):
            self.renderer.close()

    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.episode_rewards), label="A2C Cumulative Reward")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title("A2C Cumulative Reward Over Episodes")
        plt.legend()
        plt.savefig("plots/a2c_cumulative_reward.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.policy_entropies, label="A2C Policy Entropy")
        plt.xlabel("Training Step")
        plt.ylabel("Entropy")
        plt.title("A2C Training Stability")
        plt.legend()
        plt.savefig("plots/a2c_training_stability.png")
        plt.close()

def compare_models():
    """Train and compare all PG models, return the best one"""
    pygame.init()  # Initialize Pygame once at the start
    reinforce_trainer = REINFORCE(Monitor(BeehiveManagementEnv(num_hives=4, render_mode="human")))
    ppo_trainer = PPOTraining()
    a2c_trainer = A2CTraining()

    print("Training REINFORCE...")
    reinforce_model = reinforce_trainer.train()
    reinforce_results = reinforce_trainer.evaluate(model=reinforce_model)
    reinforce_trainer.record_gif(model=reinforce_model)
    reinforce_trainer.plot_results()

    print("\nTraining PPO...")
    ppo_model = ppo_trainer.train()
    ppo_results = ppo_trainer.evaluate(ppo_model)
    ppo_trainer.record_gif(ppo_model)
    ppo_trainer.plot_results()

    print("\nTraining A2C...")
    a2c_model = a2c_trainer.train()
    a2c_results = a2c_trainer.evaluate(a2c_model)
    a2c_trainer.record_gif(a2c_model)
    a2c_trainer.plot_results()

    # Close all renderers after all operations
    reinforce_trainer.close()
    ppo_trainer.close()
    a2c_trainer.close()

    results = {
        "REINFORCE": reinforce_results,
        "PPO": ppo_results,
        "A2C": a2c_results
    }

    print("\nPolicy Gradient Models Comparison:")
    best_model_name = max(results, key=lambda k: results[k]["mean_reward"])
    print(f"\nBest Model: {best_model_name}")
    for name, result in results.items():
        print(f"{name}: Mean Reward = {result['mean_reward']:.2f} ± {result['std_reward']:.2f}, "
              f"Mean Episode Length = {result['mean_length']:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(reinforce_trainer.episode_rewards), label="REINFORCE")
    plt.plot(np.cumsum(ppo_trainer.episode_rewards), label="PPO")
    plt.plot(np.cumsum(a2c_trainer.episode_rewards), label="A2C")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Policy Gradient Methods Cumulative Reward")
    plt.legend()
    plt.savefig("plots/pg_cumulative_reward.png")
    plt.close()

    return best_model_name, {
        "REINFORCE": reinforce_model,
        "PPO": ppo_model,
        "A2C": a2c_model
    }

if __name__ == "__main__":
    try:
        best_model_name, models = compare_models()
        print(f"\nBest performing model: {best_model_name}")
    finally:
        pygame.quit()  # Ensure Pygame is cleaned up