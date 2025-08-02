import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'training')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'environment')))
import gymnasium as gym
import numpy as np
import torch
from pg_training import A2CTraining
from custom_env import BeehiveManagementEnv, ActionType
from rendering import BeehiveRenderer
import pygame
from stable_baselines3 import A2C

def simulate_agent(trainer, model, model_name, episodes=3, max_steps=1000):
    """Simulate the agent running in the environment with real-time rendering and metrics"""
    env = BeehiveManagementEnv(num_hives=4, render_mode="human")
    
    # Use trainer's renderer if available, otherwise create a new one
    renderer = getattr(trainer, 'renderer', None)
    if renderer is None:
        print(f"Warning: {model_name} trainer has no renderer. Creating new BeehiveRenderer.")
        renderer = BeehiveRenderer(width=1200, height=800)
    
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            total_reward = 0

            # Debug: Log initial environment state
            hives = getattr(env, 'hives', [])
            selected_hive_id = getattr(env, 'selected_hive_id', -1)
            print(f"\n{model_name} Episode {ep+1} - Initial state:")
            print(f"Hives: {len(hives)}, Selected Hive ID: {selected_hive_id}")
            for i, hive in enumerate(hives):
                print(f"[Hive {i}] Bees: {hive.bee_population:>6} | Health: {hive.health_score:5.1f} | Food: {hive.food_stores:4.1f}kg")
            if not hives or selected_hive_id < 0 or selected_hive_id >= len(hives):
                print(f"Error: Invalid environment state in {model_name} episode {ep+1}")
                continue

            while not done and steps < max_steps:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                action_name = ActionType(action).name
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                # Render
                frame = renderer.render(env, action, reward)
                if np.mean(frame) < 10:  # Account for Colors.BACKGROUND (34, 49, 63)
                    print(f"Warning: Frame {steps} in {model_name} episode {ep+1} is nearly black (mean: {np.mean(frame):.2f})")
                
                # Log step details
                selected_hive = env.hives[env.selected_hive_id]
                print(f"\nStep {steps+1}:")
                print(f"Action: {action_name}")
                print(f"Reward: {reward:.2f} (Total: {total_reward:.2f})")
                print(f"Hive {env.selected_hive_id}: Bees: {selected_hive.bee_population:>6} "
                      f"| Health: {selected_hive.health_score:5.1f} "
                      f"| Food: {selected_hive.food_stores:4.1f}kg")
                print(f"Reward Components: { {k: f'{v:.2f}' for k, v in info['reward_components'].items()} }")
                print(f"Health Δ: {info['health_delta']:.1f}, "
                      f"Population Δ: {info['population_delta']:.0f}, "
                      f"Food Δ: {(selected_hive.food_stores - env.hives[env.selected_hive_id].food_stores):.1f}")

                steps += 1
                done = terminated or truncated
                time.sleep(0.033)  # 30 FPS for stable rendering

            # Episode summary
            print(f"\nEpisode {ep+1} Summary:")
            print(f"Steps: {steps}, Total Reward: {total_reward:.2f}")
            print(f"Final Hive States:")
            for i, hive in enumerate(env.hives):
                status = "ALIVE" if hive.bee_population > 1000 else "DEAD"
                print(f"[Hive {i}] {status:<6} | Bees: {hive.bee_population:>6} "
                      f"| Health: {hive.health_score:5.1f} | Food: {hive.food_stores:4.1f}kg")

    finally:
        if renderer is not trainer.renderer:
            renderer.close()
        env.close()

def run_best_model():
    """Load the best A2C model and simulate it in the environment"""
    pygame.init()  # Initialize Pygame
    model_name = "A2C"
    model_path = "models/pg/best_model.zip"
    
    print(f"\nLoading best model: {model_name} from {model_path}")
    
    # Initialize environment and trainer
    try:
        env = BeehiveManagementEnv(num_hives=4, render_mode="human")
        trainer = A2CTraining()  # Assumes A2CTraining initializes renderer
    except Exception as e:
        print(f"Error initializing environment or trainer: {e}")
        pygame.quit()
        return

    # Load model
    try:
        model = A2C.load(model_path, env=env)
        print(f"Successfully loaded {model_name} model from {model_path}")
    except Exception as e:
        print(f"Error loading {model_name} model from {model_path}: {e}")
        trainer.close()
        pygame.quit()
        return

    # Simulate the model
    try:
        simulate_agent(trainer, model, model_name, episodes=3, max_steps=1000)
    except Exception as e:
        print(f"Error running {model_name} model: {e}")
    
    # Cleanup
    try:
        trainer.close()
    except Exception as e:
        print(f"Error closing trainer: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    try:
        run_best_model()
        print("\nBest model simulation completed successfully!")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        pygame.quit()  # Ensure Pygame cleanup