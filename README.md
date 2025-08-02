# Raini_Benaiah_rl_summative


AsaliAsPossible
AsaliAsPossible is a reinforcement learning (RL) project that optimizes beehive management using the Proximal Policy Optimization (PPO) algorithm. The project simulates managing beehives in a custom Gymnasium environment (BeehiveManagementEnv), where an RL agent maximizes rewards by performing actions like inspecting hives, treating diseases/pests, harvesting honey, and managing resources. The PPO model, trained with Stable Baselines3, achieves a mean reward of 930.93 ± 100.95, outperforming other algorithms like A2C.
Project Overview
The goal of AsaliAsPossible is to develop an RL agent that efficiently manages beehives to maximize honey production and hive health while minimizing disease and pest impacts. Key components include:

Environment: BeehiveManagementEnv (in environment/custom_env.py) simulates four beehives with attributes like honey, health, disease, and pests.
Agent: A PPO model (models/pg/better_model.zip) trained to select optimal actions based on hive states.
Visualization: Real-time Pygame rendering (environment/rendering.py) to observe the agent’s actions.
Training: Comparison of PPO and A2C (training/compare_a2c_ppo.py), with PPO as the best performer.

Environment and Agent
BeehiveManagementEnv

Observation Space: A vector per hive (e.g., honey, health, disease, pest levels), represented as a Box space (e.g., [0.1, 0.8, 0.0, 0.2, 0.3]...).
Action Space: Discrete actions (8 total):
0: No Action
1: Inspect Hive (monitor state, small reward)
2: Treat Disease (reduce disease, avoid penalties)
3: Treat Pests (reduce pests)
4: Harvest Honey (high reward when honey is sufficient)
5: Feed Bees (maintain health/resources)
6: Expand Hive (increase capacity)
7: Reduce Hive (manage resources)


Rewards: Positive rewards for beneficial actions (e.g., 2.50–3.00 for inspecting/harvesting), penalties for high disease/pests or low health/resources.
Episode Length: ~366 steps, terminating on fixed duration or critical hive failure.

PPO Agent

Trained for 370,000 timesteps (stopped early when mean reward reached 1046.30 > 1000).
Achieves mean reward of 930.93 ± 100.95, with episode rewards of 915.20, 915.34, and 1057.78.
Uses a deterministic policy during evaluation to select optimal actions (e.g., harvesting when honey levels are high).

Setup Instructions

Clone the Repository:
git clone https://github.com/RainiBenaiah/Raini_Benaiah_rl_summative.git
cd Raini_Benaiah_rl_summative


Install Dependencies:

Ensure Python 3.8+ is installed.
Install requirements:pip install -r requirements.txt


If requirements.txt is missing, install:pip install gymnasium numpy torch stable-baselines3 pygame




Verify Project Structure:
Raini_Benaiah_rl_summative/
├── environment/
│   ├── custom_env.py
│   ├── rendering.py
├── training/
│   ├── dqn_training.py
│   ├── pg_training.py
│   ├── compare_a2c_ppo.py
├── models/
│   ├── dqn/
│   ├── pg/
│   │   ├── better_model.zip
│   │   ├── cpo_model.zip
│   │   ├── ppo_model.zip
├── main.py
├── requirements.txt
├── README.md



Running the Project
To observe the PPO agent in action:

Navigate to the project root:
cd Raini_Benaiah_rl_summative


Run main.py:
python3 main.py


What to Expect:

Console Output:
Lists model files in models/pg/ (e.g., better_model.zip).
Confirms model loading (e.g., Successfully loaded model from models/pg/better_model.zip).
Logs 3 episodes with step-by-step details:
Step number, action (e.g., “Harvest Honey”), reward (e.g., 2.50–3.00), total reward, observation (e.g., [0.1, 0.8, 0.0, 0.2, 0.3]...).
Episode summaries with total rewards (915–1057), steps (366), and time (~1.5s).




Pygame Window (1200x800):
Displays four hives with visual indicators (e.g., bars for honey/health, colors for disease/pests).
Shows actions (e.g., “Harvest Honey” text/icon) and rewards.
Updates each step to reflect hive state changes (e.g., honey bar decreases after harvesting).





Results

PPO Performance (from compare_a2c_ppo.py):
Mean Reward: 930.93 ± 100.95
Episode Length: ~366 steps
Episode Rewards: 915.20, 915.34, 1057.78
Outperforms A2C (Mean Reward: -735.17 ± 626.19)


Agent Behavior:
Maximizes rewards by:
Inspecting hives to monitor states (small rewards, e.g., 2.50).
Treating diseases/pests to avoid penalties.
Harvesting honey at optimal times (high rewards, e.g., 3.00).
Feeding/expanding hives to maintain health/resources.


Best episode (1057.78) demonstrates effective action sequencing for high honey production and hive health.



Project Structure

environment/custom_env.py: Defines BeehiveManagementEnv with observation/action spaces and reward logic.
environment/rendering.py: Implements BeehiveRenderer for Pygame visualization.
training/compare_a2c_ppo.py: Trains and compares A2C and PPO, saving models to models/pg/.
training/dqn_training.py: Trains DQN models (saved to models/dqn/).
training/pg_training.py: Trains policy gradient models (e.g., PPO).
main.py: Loads PPO model (better_model.zip) and runs 3 episodes, logging actions and rewards.
models/pg/: Contains PPO models (better_model.zip, cpo_model.zip, ppo_model.zip).
requirements.txt: Lists project dependencies.
You can also try different models saved in the models folder and compare them.
Troubleshooting

Model Loading Error:

Verify model files:ls models/pg


If missing, re-run training:python3 training/compare_a2c_ppo.py


Update model_paths in main.py if paths differ.


No Pygame Visuals:

Ensure Pygame is installed:pip install pygame


Verify render_mode="human" in custom_env.py.


Low Rewards:

If rewards are low (<900), re-run compare_a2c_ppo.py with:"ent_coef": 0.05  # In PPOTraining





Contact
For issues, contact Raini Benaiah (b.raini@alustudent.com
