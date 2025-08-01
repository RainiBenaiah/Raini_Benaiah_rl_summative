#!/usr/bin/env python3
"""
Demo script showing random agent taking actions in the beehive management environment.
This demonstrates the visualization without any learning/training involved.
"""

import pygame
import numpy as np
import time
import os
import sys
from typing import Dict, List
import imageio
from custom_env import BeehiveManagementEnv, ActionType
from rendering import BeehiveRenderer

class RandomAgentDemo:
    """Demonstration of random agent in beehive environment"""
    
    def __init__(self, num_hives: int = 5, record_gif: bool = True):
        self.num_hives = num_hives
        self.record_gif = record_gif
        self.frames = []  # For GIF recording
        
        # Initialize environment and renderer
        self.env = BeehiveManagementEnv(num_hives=num_hives, render_mode="human")
        self.renderer = BeehiveRenderer(width=1200, height=800)
        
        # Demo statistics
        self.action_counts = {action_type.name: 0 for action_type in ActionType}
        self.total_steps = 0
        self.episodes_completed = 0
        
        print(" Beehive Management RL Environment Demo")
        print("=" * 50)
        print(f"Environment: {num_hives} hives, {self.env.max_steps} steps per episode")
        print(f"Action Space: {self.env.action_space.n} discrete actions")
        print(f"Observation Space: {self.env.observation_space.shape}")
        print("=" * 50)
    
    def print_action_analysis(self, action: int, reward: float, info: Dict):
        """Print detailed analysis of the action taken"""
        action_type = ActionType(action)
        action_category = self._get_action_category(action)
        
        season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
        season_value = info.get('season', 0)
        season_index = min(3, max(0, season_value))  # Ensure valid index
        
        print(f"\n Step {self.total_steps} Analysis:")
        print(f"   Action: {action_type.name.replace('_', ' ')}")
        print(f"   Category: {action_category}")
        print(f"   Target Hive: {info.get('selected_hive', 'N/A')}")
        print(f"   Reward: {reward:.2f}")
        print(f"   Season: {season_names[season_index]}")
        print(f"   Active Hives: {info['active_hives']}/{self.num_hives}")
        print(f"   Total Production: {info.get('honey_production', 0.0):.1f}kg")
        
        # Update statistics
        self.action_counts[action_type.name] += 1
    
    def _get_action_category(self, action: int) -> str:
        """Categorize action for analysis"""
        if action <= 9:
            return " LOCATION MANAGEMENT"
        elif action <= 19:
            return " HIVE MAINTENANCE"
        elif action <= 29:
            return " POPULATION MANAGEMENT"
        elif action <= 39:
            return " ENVIRONMENTAL ADAPTATION"
        elif action <= 49:
            return " MONITORING & ANALYTICS"
        elif action <= 59:
            return " RESOURCE MANAGEMENT"
        elif action <= 69:
            return " EMERGENCY ACTIONS"
        else:
            return " RESEARCH & DEVELOPMENT"

    def demonstrate_action_space(self):
        """Demonstrate all action categories with examples"""
        print("\n Action Space Demonstration:")
        print("=" * 50)
        
        # Sample actions from each category
        category_examples = {
            "Location Management (0-9)": [0, 1, 2, 3, 4],
            "Hive Maintenance (10-19)": [10, 13, 15, 17, 19],
            "Population Management (20-29)": [20, 21, 22, 26, 28],
            "Environmental Adaptation (30-39)": [30, 32, 34, 36, 38],
            "Monitoring & Analytics (40-49)": [40, 42, 45, 47, 49],
            "Resource Management (50-59)": [50, 52, 54, 56, 58],
            "Emergency Actions (60-69)": [60, 62, 65, 67, 69],
            "Research & Development (70-79)": [70, 72, 75, 77, 79]
        }
        
        for category, actions in category_examples.items():
            print(f"\n{category}:")
            for action in actions:
                action_name = ActionType(action).name.replace('_', ' ').title()
                print(f"  {action:2d}: {action_name}")
        
        print(f"\nTotal Actions Available: {len(ActionType)} (Discrete Action Space)")
    
    def run_episode(self, max_steps: int = 100, step_delay: float = 0.1) -> Dict:
        """Run one episode with random actions"""
        obs, info = self.env.reset()
        episode_reward = 0
        episode_actions = []
        
        print(f"\n🎬 Starting Episode {self.episodes_completed + 1}")
        print("-" * 30)
        
        for step in range(max_steps):
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return {"early_termination": True}
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return {"early_termination": True}
                    elif event.key == pygame.K_SPACE:
                        input("Press Enter to continue...")
            
            # Take random action
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Record action and reward
            episode_reward += reward
            episode_actions.append(action)
            self.total_steps += 1
            
            # Print action analysis
            self.print_action_analysis(action, reward, info)
            
            # Render environment
            frame = self.renderer.render(self.env, action, reward)
            
            # Record frame for GIF
            if self.record_gif:
                # Convert pygame surface to RGB array
                rgb_frame = pygame.surfarray.array3d(self.renderer.screen)
                rgb_frame = np.transpose(rgb_frame, (1, 0, 2))
                self.frames.append(rgb_frame)
            
            # Add delay for visualization
            time.sleep(step_delay)
            
            # Check termination
            if terminated or truncated:
                break
        
        self.episodes_completed += 1
        
        return {
            "episode_reward": episode_reward,
            "episode_actions": episode_actions,
            "steps_taken": step,
            "early_termination": False
        }
    
    def print_statistics(self):
        """Print comprehensive statistics about actions taken"""
        print("\n DEMONSTRATION STATISTICS")
        print("=" * 60)
        print(f"Total Steps: {self.total_steps}")
        print(f"Episodes Completed: {self.episodes_completed}")
        print(f"Average Steps per Episode: {self.total_steps / max(1, self.episodes_completed):.1f}")
        
        # Action category statistics
        category_counts = {}
        for action_name, count in self.action_counts.items():
            if count > 0:
                category = self._get_action_category_from_name(action_name)
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += count
        
        print(f"\n Actions by Category:")
        total_actions = sum(category_counts.values())
        for category, count in sorted(category_counts.items()):
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            print(f"  {category}: {count} actions ({percentage:.1f}%)")
        
        # Most used actions
        print(f"\n Top 10 Most Used Actions:")
        sorted_actions = sorted(self.action_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (action_name, count) in enumerate(sorted_actions[:10]):
            if count > 0:
                action_display = action_name.replace('_', ' ').title()
                print(f"  {i+1:2d}. {action_display}: {count} times")
        
        # Action space coverage
        used_actions = sum(1 for count in self.action_counts.values() if count > 0)
        total_actions_available = len(ActionType)
        coverage = (used_actions / total_actions_available * 100)
        print(f"\n Action Space Coverage: {used_actions}/{total_actions_available} ({coverage:.1f}%)")
    
    def _get_action_category_from_name(self, action_name: str) -> str:
        """Get action category from action name"""
        # Convert action name back to number to categorize
        try:
            action_num = ActionType[action_name].value
            return self._get_action_category(action_num)
        except:
            return " UNKNOWN"
    
    def save_gif(self, filename: str = "beehive_demo.gif", fps: int = 10):
        """Save recorded frames as GIF"""
        if not self.frames:
            print(" No frames recorded for GIF creation")
            return
        
        try:
            print(f" Creating GIF with {len(self.frames)} frames...")
            
            # Resize frames if too large (for file size optimization)
            processed_frames = []
            for frame in self.frames[::2]:  # Take every other frame to reduce size
                # Convert to uint8 if needed
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                processed_frames.append(frame)
            
            # Save as GIF
            imageio.mimsave(filename, processed_frames, fps=fps, loop=0)
            print(f" GIF saved as '{filename}' ({len(processed_frames)} frames, {fps} FPS)")
            
            # Print file info
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f" File size: {file_size:.1f} MB")
            
        except Exception as e:
            print(f" Error creating GIF: {e}")
    
    def run_comprehensive_demo(self, num_episodes: int = 2, steps_per_episode: int = 50):
        """Run comprehensive demonstration"""
        print(" Starting Comprehensive Random Agent Demonstration")
        print("=" * 60)
        
        # Show action space
        self.demonstrate_action_space()
        
        print(f"\n Demo Configuration:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Steps per Episode: {steps_per_episode}")
        print(f"  Recording GIF: {'Yes' if self.record_gif else 'No'}")
        print(f"  Hives: {self.num_hives}")
        
        print(f"\n Controls:")
        print(f"  ESC: Quit demo")
        print(f"  SPACE: Pause (press Enter to continue)")
        print(f"  Close window: Exit")
        
        input("\nPress Enter to start demo...")
        
        # Run episodes
        for episode in range(num_episodes):
            result = self.run_episode(steps_per_episode, step_delay=0.15)
            
            if result.get("early_termination"):
                print("\n Demo stopped by user")
                break

            print(f"\n Episode {episode + 1} Complete:")
            print(f"   Total Reward: {result['episode_reward']:.2f}")
            print(f"   Steps Taken: {result['steps_taken']}")
            print(f"   Actions Used: {len(set(result['episode_actions']))}")
            
            if episode < num_episodes - 1:
                print("\n Preparing next episode...")
                time.sleep(1)
        
        # Show final statistics
        self.print_statistics()
        
        # Save GIF if recording
        if self.record_gif and self.frames:
            self.save_gif("beehive_random_agent_demo.gif", fps=8)
        
        print("\n Demo Complete!")
        print(f"   Total demonstration time: ~{self.total_steps * 0.15:.1f} seconds")
        print(f"   Thank you for watching the Beehive Management RL Environment!")
    
    def close(self):
        """Clean up resources"""
        self.renderer.close()
        self.env.close()

def main():
    """Main demo function"""
    print(" Beehive Management RL Environment - Random Agent Demo")
    print(" This demonstrates the environment with random actions (no learning)")
    print("-" * 70)
    
    # Configuration
    config = {
        "num_hives": 4,           # Number of hives in environment
        "num_episodes": 2,        # Number of episodes to run
        "steps_per_episode": 60,  # Steps per episode
        "record_gif": True        # Whether to record GIF
    }
    
    print(" Configuration:")
    for key, value in config.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Check dependencies
    try:
        import imageio
        print(" All dependencies available")
    except ImportError:
        print("  imageio not available - GIF recording disabled")
        config["record_gif"] = False
    
    try:
        # Initialize demo
        demo = RandomAgentDemo(
            num_hives=config["num_hives"],
            record_gif=config["record_gif"]
        )
        
        # Run comprehensive demonstration
        demo.run_comprehensive_demo(
            num_episodes=config["num_episodes"],
            steps_per_episode=config["steps_per_episode"]
        )
        
    except KeyboardInterrupt:
        print("\n Demo interrupted by user")
    except Exception as e:
        print(f"\n Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        try:
            demo.close()
        except:
            pass

        print("\n Goodbye!")

if __name__ == "__main__":
    main()