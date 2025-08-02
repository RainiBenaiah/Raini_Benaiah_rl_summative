import gymnasium as gym
import numpy as np
import imageio
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from custom_env import BeehiveManagementEnv
from rendering import BeehiveRenderer
from dqn_training import DQNTraining
from pg_training import REINFORCE, PPOTraining, A2CTraining
import os

def record_episode(model, env, renderer, filename="episode.mp4", max_steps=1000, model_type="sb3"):
    """Record a single episode video"""
    frames = []
    obs, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < max_steps:
        if model_type == "reinforce":
            obs_tensor = torch.FloatTensor(obs).to(model.device)
            action_probs = model.policy(obs_tensor)
            action = Categorical(action_probs).sample().item()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frame = renderer.render(env, action, reward)
        frames.append(np.transpose(frame, (1, 0, 2)))
        steps += 1
        done = terminated or truncated

    writer = imageio.get_writer(filename, fps=30, macro_block_size=1)
    for frame in frames:
        writer.append_data(frame.astype(np.uint8))
    writer.close()
    print(f"Video saved as {filename}, Total Reward: {total_reward:.2f}")

def main():
    """Main entry point for running RL experiments"""
    env = Monitor(BeehiveManagementEnv(num_hives=4, render_mode="human"))
    renderer = BeehiveRenderer(width=1200, height=800)

    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("models/pg", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Train models
    print("Training DQN...")
    dqn_trainer = DQNTraining()
    dqn_model = dqn_trainer.train()
    dqn_results = dqn_trainer.evaluate(dqn_model)
    dqn_trainer.record_gif(dqn_model)

    print("\nTraining Policy Gradient Models...")
    best_pg_name, pg_models = compare_models()

    # Evaluate models
    ppo_results = evaluate_model(pg_models["PPO"], env)
    a2c_results = evaluate_model(pg_models["A2C"], env)
    reinforce_results = REINFORCE(Monitor(BeehiveManagementEnv(num_hives=4, render_mode="human"))).evaluate(pg_models["REINFORCE"])

    # Compare results
    results = {
        "DQN": dqn_results,
        "PPO": ppo_results,
        "A2C": a2c_results,
        "REINFORCE": reinforce_results
    }

    print("\n=== Model Comparison ===")
    best_model_name = max(results, key=lambda k: results[k]["mean_reward"])
    print(f"Best Model: {best_model_name}")
    for name, result in results.items():
        print(f"{name}: Mean Reward = {result['mean_reward']:.2f} ± {result['std_reward']:.2f}, "
              f"Mean Episode Length = {result['mean_length']:.2f}")

    # Plot combined cumulative rewards
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(dqn_trainer.episode_rewards), label="DQN")
    plt.plot(np.cumsum(PPOTraining().episode_rewards), label="PPO")
    plt.plot(np.cumsum(A2CTraining().episode_rewards), label="A2C")
    plt.plot(np.cumsum(REINFORCE(Monitor(BeehiveManagementEnv(num_hives=4))).episode_rewards), label="REINFORCE")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("All Models Cumulative Reward")
    plt.legend()
    plt.savefig("plots/all_models_cumulative_reward.png")
    plt.close()

    # Record videos for best model
    print(f"\nRecording videos for best model ({best_model_name})...")
    best_model = dqn_model if best_model_name == "DQN" else pg_models[best_model_name]
    model_type = "reinforce" if best_model_name == "REINFORCE" else "sb3"
    for i in range(3):
        record_episode(best_model, env, renderer, f"videos/best_model_{best_model_name}_episode_{i+1}.mp4", model_type=model_type)

    renderer.close()
    env.close()

if __name__ == "__main__":
    main()