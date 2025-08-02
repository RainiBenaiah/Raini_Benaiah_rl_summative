
# AsaliAsPossible - Beehive Management with Reinforcement Learning

**Project by:** Raini Benaiah  
**Email:** b.raini@alustudent.com

AsaliAsPossible is a reinforcement learning project focused on optimizing beehive management using the Proximal Policy Optimization (PPO) algorithm. The project simulates hive dynamics in a custom Gymnasium environment, where an RL agent learns to maximize honey production, maintain hive health, and mitigate diseases and pests.

## Project Overview

The aim is to train an agent that takes intelligent actions such as inspecting hives, treating issues, harvesting honey, and managing resources effectively.

### Key Features

- **Environment:** `BeehiveManagementEnv` simulates four hives with state features like honey levels, health, disease, and pest presence.
- **Agent:** A PPO model trained using Stable Baselines3 to act optimally.
- **Visualization:** Real-time rendering using Pygame for interactive hive monitoring.
- **Training Comparison:** PPO vs A2C algorithms.

## Environment and Agent Details

### BeehiveManagementEnv

- **Observation Space:** Vector of hive attributes (e.g., `[0.1, 0.8, 0.0, 0.2, 0.3]`).
- **Action Space (8 Discrete Actions):**
  0. No Action  
  1. Inspect Hive  
  2. Treat Disease  
  3. Treat Pests  
  4. Harvest Honey  
  5. Feed Bees  
  6. Expand Hive  
  7. Reduce Hive

- **Rewards:**  
  - +2.5–3.0 for useful actions  
  - Penalties for high disease, pest infestation, and poor hive health

- **Episode Length:** ~366 steps

### PPO Agent

- **Training Steps:** 370,000  
- **Performance:**  
  - Mean Reward: 930.93 ± 100.95  
  - Best Episode: 1057.78
- **Evaluation:** Deterministic policy to select optimal actions

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/RainiBenaiah/Raini_Benaiah_rl_summative.git
cd Raini_Benaiah_rl_summative
```

### 2. Install Dependencies

Ensure you have Python 3.8+

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing:

```bash
pip install gymnasium numpy torch stable-baselines3 pygame
```

## Project Structure

```
Raini_Benaiah_rl_summative/
├── environment/
│   ├── custom_env.py           # Environment logic
│   ├── rendering.py            # Pygame rendering
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
├── main.py                     # Main evaluation script
├── requirements.txt
├── README.md
```

## Running the Project

To observe the PPO agent in action:

```bash
cd Raini_Benaiah_rl_summative
python3 main.py
```

### What You’ll See

- **Console Logs:**
  - Step-wise actions with rewards
  - Episode summaries (reward, steps, time)

- **Pygame Window (1200x800):**
  - 4 hives with visual indicators
  - Actions and rewards shown
  - Real-time hive state updates

## Results
 ## There are other models like DQN ans Reinforcement but this two were outstanding
### PPO vs A2C Performance

| Metric         | PPO                    | A2C                   |
|----------------|------------------------|------------------------|
| Mean Reward    | 930.93 ± 100.95        | -735.17 ± 626.19      |
| Best Episode   | 1057.78                | N/A                   |
| Episode Length | ~366 steps             | ~366 steps            |

### Agent Behavior Insights

- Inspects hives for low-reward monitoring
- Treats diseases/pests promptly
- Harvests only when honey is abundant
- Maintains hive strength via feeding and expansion

## Troubleshooting

- **Model Load Errors:**
  ```bash
  ls models/pg
  python3 training/compare_a2c_ppo.py
  ```

- **No Visuals (Pygame):**
  ```bash
  pip install pygame
  ```
  Ensure `render_mode="human"` is set in `custom_env.py`.

- **Low Rewards (<900)?**
  Modify PPO training hyperparameters, for example:
  ```python
  "ent_coef": 0.05
  ```

## Contact

For questions or contributions, reach out to:

**Raini Benaiah**  
Email: b.raini@alustudent.com


