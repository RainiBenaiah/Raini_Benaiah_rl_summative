import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from custom_env import BeehiveManagementEnv

def train_dqn(save_path="models/dqn/dqn_model"):
    """Train DQN model on BeehiveManagementEnv"""
    # Create and wrap environment
    env = BeehiveManagementEnv(num_hives=4)
    env = Monitor(env)

    # Define evaluation environment
    eval_env = BeehiveManagementEnv(num_hives=4)
    eval_env = Monitor(eval_env)

    # Create directory for saving models
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Define hyperparameters
    hyperparams = {
        "learning_rate": 1e-3,
        "buffer_size": 100000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.02,
        "learning_starts": 1000,
        "target_update_interval": 1000,
        "train_freq": 4,
    }

    # Initialize DQN model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        **hyperparams,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    # Define callbacks
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

    # Train model
    model.learn(
        total_timesteps=100000,
        callback=eval_callback,
        progress_bar=True
    )

    # Save the final model
    model.save(save_path)
    print(f"DQN model saved to {save_path}")

    return model

def evaluate_dqn(model, env, n_episodes=5):
    """Evaluate trained DQN model"""
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

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "episode_rewards": episode_rewards
    }

if __name__ == "__main__":
    # Train DQN model
    dqn_model = train_dqn()
    
    # Evaluate model
    eval_env = BeehiveManagementEnv(num_hives=4)
    results = evaluate_dqn(dqn_model, eval_env)
    
    print("\nDQN Evaluation Results:")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.2f}")