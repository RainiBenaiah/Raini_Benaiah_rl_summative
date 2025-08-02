import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'environment')))
from custom_env import BeehiveManagementEnv
from rendering import BeehiveRenderer
import pygame
import time

def run_best_ppo_model():
    pygame.init()
    env = BeehiveManagementEnv(num_hives=4, render_mode="human")
    renderer = BeehiveRenderer(width=1200, height=800)
    
    model_name = "PPO"
    model_paths = [
        "training/models/pg/cpo_model",  # Best model from comparison(ppo vs a2c)
       # "training/models/pg/c_model",    # From PPO output
        #"training/models/pg/bet_model"     # Default PPOTraining path
    ]
    
    # Debug: List files in training/models/pg
    models_pg_path = "training/models/pg"
    print(f"Checking directory: {models_pg_path}")
    if os.path.exists(models_pg_path):
        print("Files found:")
        for file in os.listdir(models_pg_path):
            full_path = os.path.join(models_pg_path, file)
            print(f"  - {file} (Full path: {full_path})")
    else:
        print(f"Directory {models_pg_path} does not exist")
    
    # Try loading model from each path
    model = None
    for path in model_paths:
        print(f"\nTrying to load {model_name} from {path}.zip")
        print(f"File exists: {os.path.exists(path + '.zip')}")
        try:
            model = PPO.load(path, env=env)
            print(f"Successfully loaded model from {path}.zip")
            break
        except Exception as e:
            print(f"Error loading PPO model from {path}.zip: {str(e)}")
    
    if model is None:
        print("Failed to load any model. Exiting.")
        return
    
    # Action mapping (based on typical BeehiveManagementEnv actions)
    action_names = {
        0: "No Action",
        1: "Inspect Hive",
        2: "Treat Disease",
        3: "Treat Pests",
        4: "Harvest Honey",
        5: "Feed Bees",
        6: "Expand Hive",
        7: "Reduce Hive"
    }
    
    # Run 3 episodes
    n_episodes = 3
    max_steps = 1000
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        print(f"\nEpisode {episode+1}:")
        print("Step | Action | Reward | Total Reward | Observation")
        print("-" * 60)
        
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            action_name = action_names.get(int(action), f"Unknown Action ({action})")
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            frame = renderer.render(env.unwrapped, action, reward)
            steps += 1
            done = terminated or truncated
            
            elapsed_time = time.time() - start_time
            # Truncate observation for readability
            obs_str = f"{obs[:5]}..." if len(obs) > 5 else str(obs)
            print(f"{steps:4d} | {action_name:14} | {reward:6.2f} | {total_reward:12.2f} | {obs_str}")
        
        print(f"\nEpisode {episode+1} Summary:")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps: {steps}")
        print(f"Time Elapsed: {elapsed_time:.2f}s")
    
    env.close()
    renderer.close()
    pygame.quit()
    
    print("\nPPO best model simulation completed successfully!")

if __name__ == "__main__":
    run_best_ppo_model()