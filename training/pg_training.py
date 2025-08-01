import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from custom_env import BeehiveManagementEnv
import os

class REINFORCE:
    def __init__(self, env, learning_rate=1e-3, gamma=0.99):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.model = None  # To be implemented with custom policy gradient

    def train(self, total_timesteps=100000):
        # Placeholder for REINFORCE implementation
        print("REINFORCE training not implemented in Stable Baselines3. Using custom implementation.")
        # You would need to implement a custom REINFORCE algorithm here
        pass

def train_ppo(save_path="models/pg/ppo_model"):
    """Train PPO model on BeehiveManagementEnv"""
    env = BeehiveManagementEnv(num_hives=4)
    env = Monitor(env)

    eval_env = BeehiveManagementEnv(num_hives=4)
    eval_env = Monitor(eval_env)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
        env=env,
        **hyperparams,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path),
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

    model.save(save_path)
    print(f"PPO model saved to {save_path}")

    return model

def train_a2c(save_path="models/pg/a2c_model"):
    """Train A2C model on BeehiveManagementEnv"""
    env = BeehiveManagementEnv(num_hives=4)
    env = Monitor(env)

    eval_env = BeehiveManagementEnv(num_hives=4)
    eval_env = Monitor(eval_env)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
        env=env,
        **hyperparams,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path),
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

    model.save(save_path)
    print(f"A2C model saved to {save_path}")

    return model

def evaluate_model(model, env, n_episodes=5):
    """Evaluate trained model"""
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "episode_rewards": episode_rewards
    }

if __name__ == "__main__":
    # Train PPO
    ppo_model = train_ppo()
    eval_env = BeehiveManagementEnv(num_hives=4)
    ppo_results = evaluate_model(ppo_model, eval_env)
    print("\nPPO Evaluation Results:")
    print(f"Mean Reward: {ppo_results['mean_reward']:.2f} ± {ppo_results['std_reward']:.2f}")
    print(f"Mean Episode Length: {ppo_results['mean_length']:.2f}")

    # Train A2C
    a2c_model = train_a2c()
    a2c_results = evaluate_model(a2c_model, eval_env)
    print("\nA2C Evaluation Results:")
    print(f"Mean Reward: {a2c_results['mean_reward']:.2f} ± {a2c_results['std_reward']:.2f}")
    print(f"Mean Episode Length: {a2c_results['mean_length']:.2f}")

    # Note: REINFORCE is not implemented in Stable Baselines3
    # You would need a custom implementation for REINFORCE
    print("\nNote: REINFORCE training requires custom implementation.")