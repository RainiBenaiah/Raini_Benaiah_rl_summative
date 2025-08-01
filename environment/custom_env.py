import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import random
from enum import IntEnum
from dataclasses import dataclass
import dataclasses
import math
import pygame
import time

class ActionType(IntEnum):
    """Discrete action enumeration for beehive management"""
    # Location Management (0-9)
    KEEP_CURRENT_LOCATION = 0
    MOVE_TO_HIGH_NECTAR_ZONE = 1
    MOVE_TO_PROTECTED_AREA = 2
    MOVE_TO_WATER_SOURCE = 3
    MOVE_TO_OPTIMAL_TEMP_ZONE = 4
    RETURN_TO_BASE_LOCATION = 5
    SEASONAL_MIGRATION_NORTH = 6
    SEASONAL_MIGRATION_SOUTH = 7
    MOVE_TO_SHELTER_AREA = 8
    RANDOM_EXPLORATION_MOVE = 9
    
    # Hive Maintenance (10-19)
    SCHEDULE_IMMEDIATE_INSPECTION = 10
    SCHEDULE_WEEKLY_INSPECTION = 11
    SCHEDULE_MONTHLY_INSPECTION = 12
    PROVIDE_SUGAR_FEEDING = 13
    PROVIDE_POLLEN_SUBSTITUTE = 14
    APPLY_DISEASE_TREATMENT = 15
    APPLY_PEST_CONTROL = 16
    CLEAN_HIVE_ENTRANCE = 17
    REPLACE_OLD_FRAMES = 18
    ADD_HONEY_SUPERS = 19
    
    # Population Management (20-29)
    RECOMMEND_HIVE_SPLITTING = 20
    RECOMMEND_HIVE_MERGING = 21
    INTRODUCE_NEW_QUEEN = 22
    REPLACE_OLD_QUEEN = 23
    ADD_NURSE_BEES = 24
    REMOVE_EXCESS_DRONES = 25
    MONITOR_SWARMING_BEHAVIOR = 26
    PREVENT_SWARMING = 27
    ENCOURAGE_BROOD_PRODUCTION = 28
    QUARANTINE_SICK_BEES = 29
    
    # Environmental Adaptation (30-39)
    INSTALL_SHADE_PROTECTION = 30
    INSTALL_WIND_BARRIER = 31
    IMPROVE_VENTILATION = 32
    ADD_ENTRANCE_REDUCER = 33
    INSULATE_FOR_WINTER = 34
    REMOVE_WINTER_INSULATION = 35
    ADJUST_HIVE_ORIENTATION = 36
    RAISE_HIVE_HEIGHT = 37
    LOWER_HIVE_HEIGHT = 38
    CAMOUFLAGE_HIVE = 39
    
    # Monitoring and Analytics (40-49)
    DEPLOY_SOUND_MONITORING = 40
    INCREASE_MONITORING_FREQUENCY = 41
    DECREASE_MONITORING_FREQUENCY = 42
    CALIBRATE_SENSORS = 43
    UPDATE_TRACKING_SYSTEM = 44
    ANALYZE_FLIGHT_PATTERNS = 45
    RECORD_BEHAVIORAL_DATA = 46
    MEASURE_HIVE_WEIGHT = 47
    CHECK_HONEY_MOISTURE = 48
    ASSESS_COMB_QUALITY = 49
    
    # Resource Management (50-59)
    HARVEST_HONEY = 50
    EXTRACT_EXCESS_HONEY = 51
    LEAVE_HONEY_FOR_WINTER = 52
    REDISTRIBUTE_HONEY_STORES = 53
    COLLECT_PROPOLIS = 54
    COLLECT_BEESWAX = 55
    MANAGE_POLLEN_STORES = 56
    OPTIMIZE_STORAGE_SPACE = 57
    PREPARE_EMERGENCY_FEED = 58
    CONSERVE_RESOURCES = 59
    
    # Emergency Actions (60-69)
    EMERGENCY_RELOCATION = 60
    EMERGENCY_FEEDING = 61
    EMERGENCY_TREATMENT = 62
    EMERGENCY_QUEEN_REPLACEMENT = 63
    EMERGENCY_HIVE_SPLITTING = 64
    EMERGENCY_PEST_REMOVAL = 65
    EMERGENCY_DISEASE_QUARANTINE = 66
    EMERGENCY_WEATHER_PROTECTION = 67
    EMERGENCY_PREDATOR_DEFENSE = 68
    EMERGENCY_HIVE_RESCUE = 69
    
    # Research and Development (70-79)
    EXPERIMENT_NEW_LOCATION = 70
    TEST_NEW_FEEDING_METHOD = 71
    TRIAL_DISEASE_PREVENTION = 72
    IMPLEMENT_NEW_MONITORING = 73
    VALIDATE_PRODUCTION_METHOD = 74
    COMPARE_HIVE_PERFORMANCE = 75
    BENCHMARK_AGAINST_STANDARDS = 76
    DOCUMENT_BEST_PRACTICES = 77
    SHARE_KNOWLEDGE_NETWORK = 78
    CONTINUOUS_IMPROVEMENT = 79

@dataclass
class HiveState:
    """Represents the state of a single beehive"""
    hive_id: int
    location: Tuple[float, float]  # (x, y) coordinates
    honey_production: float  # kg per month
    bee_population: int
    health_score: float  # 0-100
    queen_age: int  # months
    queen_present: bool
    disease_level: float  # 0-1
    pest_level: float  # 0-1
    food_stores: float  # kg
    last_inspection: int  # days ago
    hive_age: int  # months
    productivity_trend: float  # -1 to 1
    sound_activity: float  # 0-1
    temperature: float  # Celsius
    humidity: float  # 0-1
    nectar_availability: float  # 0-1
    season: int  # 0-3 (spring, summer, autumn, winter)
    weather_risk: float  # 0-1
    predator_risk: float  # 0-1

class BeehiveManagementEnv(gym.Env):
    """
    Custom Gymnasium environment for beehive management using reinforcement learning.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: Optional[str] = None, num_hives: int = 5):
        super().__init__()
        
        self.render_mode = render_mode
        self.num_hives = num_hives
        self.max_steps = 365  # One year simulation
        self.current_step = 0
        
        # Environment boundaries (100x100 grid representing the apiary area)
        self.world_size = 100
        self.base_location = (50, 50)
        
        # Define action space - DISCRETE actions (80 possible actions)
        self.action_space = spaces.Discrete(80)
        
        # Define observation space - continuous values for each hive state
        # Each hive has 20 state variables
        obs_low = np.array([
            # For each hive: [location_x, location_y, honey_production, bee_population, 
            # health_score, queen_age, queen_present, disease_level, pest_level,
            # food_stores, last_inspection, hive_age, productivity_trend, sound_activity,
            # temperature, humidity, nectar_availability, season, weather_risk, predator_risk]
            *([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -20, 0, 0, 0, 0, 0] * num_hives),
            0  # current_step
        ])
        
        obs_high = np.array([
            # For each hive max values
            *([100, 100, 50, 100000, 100, 60, 1, 1, 1, 100, 365, 120, 1, 1, 50, 1, 1, 3, 1, 1] * num_hives),
            365  # max_steps
        ])
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Initialize hives
        self.hives: List[HiveState] = []
        self.selected_hive_id = 0  # Which hive the action applies to
        
        # Reward tracking
        self.total_reward = 0
        self.episode_honey_production = 0
        self.episode_hive_losses = 0
        
        # Environmental factors
        self.seasonal_multiplier = [0.6, 1.0, 0.8, 0.3]  # Spring, Summer, Autumn, Winter
        self.nectar_zones = self._generate_nectar_zones()
        self.water_sources = [(20, 30), (70, 80), (40, 10)]
        
        self.reset()
    
    def _generate_nectar_zones(self) -> List[Tuple[float, float, float]]:
        """Generate nectar-rich zones with (x, y, richness) values"""
        zones = []
        for _ in range(8):
            x = random.uniform(10, 90)
            y = random.uniform(10, 90)
            richness = random.uniform(0.3, 1.0)
            zones.append((x, y, richness))
        return zones
    
    def _create_random_hive(self, hive_id: int) -> HiveState:
        """Create a hive with randomized initial state"""
        return HiveState(
            hive_id=hive_id,
            location=(random.uniform(20, 80), random.uniform(20, 80)),
            honey_production=random.uniform(2, 8),
            bee_population=random.randint(20000, 80000),
            health_score=random.uniform(60, 95),
            queen_age=random.randint(1, 24),
            queen_present=random.choice([True, True, True, False]),  # 75% chance
            disease_level=random.uniform(0, 0.3),
            pest_level=random.uniform(0, 0.4),
            food_stores=random.uniform(5, 25),
            last_inspection=random.randint(0, 30),
            hive_age=random.randint(1, 36),
            productivity_trend=random.uniform(-0.5, 0.5),
            sound_activity=random.uniform(0.3, 0.9),
            temperature=random.uniform(15, 35),
            humidity=random.uniform(0.4, 0.8),
            nectar_availability=random.uniform(0.2, 0.8),
            season=min(3, max(0, self.current_step // 91)),  # Clamp to valid range 0-3
            weather_risk=random.uniform(0, 0.6),
            predator_risk=random.uniform(0, 0.3)
        )
    
    def _get_observation(self) -> np.ndarray:
        """Convert current state to observation array"""
        obs = []
        for hive in self.hives:
            obs.extend([
                hive.location[0], hive.location[1], hive.honey_production,
                hive.bee_population / 1000,  # Scale down
                hive.health_score, hive.queen_age, float(hive.queen_present),
                hive.disease_level, hive.pest_level, hive.food_stores,
                hive.last_inspection, hive.hive_age, hive.productivity_trend,
                hive.sound_activity, hive.temperature, hive.humidity,
                hive.nectar_availability, hive.season, hive.weather_risk,
                hive.predator_risk
            ])
        obs.append(self.current_step)
        return np.array(obs, dtype=np.float32)
    def _calculate_reward(self, action: int, hive: HiveState, prev_hive: HiveState) -> float:
        """Balanced multi-dimensional reward function"""
        reward = 0
        
        # 1. Core Viability (30%)
        health_norm = hive.health_score / 100
        population_norm = hive.bee_population / 80000  
        core_reward = 0.5 * health_norm + 0.5 * population_norm
        reward += 3 * core_reward
        
        # 2. Productivity (25%)
        honey_change = hive.honey_production - prev_hive.honey_production
        productivity = np.clip(hive.productivity_trend, -1, 1)
        prod_reward = (0.7 * np.tanh(honey_change) + 0.3 * productivity)
        reward += 2.5 * prod_reward
        
        # 3. Action Efficiency (20%)
        action_score = self._evaluate_action_effectiveness(ActionType(action), hive, prev_hive)
        reward += 2 * action_score
        
        # 4. Risk Mitigation (15%)
        risk_penalty = (hive.disease_level + hive.pest_level + 0.5*hive.predator_risk)
        reward -= 1.5 * risk_penalty
        
        # 5. Sustainability (10%)
        food_balance = np.log1p(hive.food_stores) / 5
        queen_bonus = 0.2 if hive.queen_present else 0
        reward += food_balance + queen_bonus
        
        return float(np.clip(reward, -5, 10))
    
    def _get_nectar_score(self, location: Tuple[float, float]) -> float:
        """Calculate nectar availability score for a location"""
        max_score = 0
        for zone_x, zone_y, richness in self.nectar_zones:
            distance = math.sqrt((location[0] - zone_x)**2 + (location[1] - zone_y)**2)
            score = richness * max(0, 1 - distance / 20)  # 20 unit effective radius
            max_score = max(max_score, score)
        return max_score
    
    def _evaluate_action_effectiveness(self, action: ActionType, hive: HiveState, prev_hive: HiveState) -> float:
        """Context-aware action scoring"""
        score = 0.3  # Neutral baseline
        
        # Location Actions
        if action in [ActionType.MOVE_TO_HIGH_NECTAR_ZONE, ActionType.MOVE_TO_WATER_SOURCE]:
            nectar_gain = self._get_nectar_score(hive.location) - self._get_nectar_score(prev_hive.location)
            score = 0.5 + nectar_gain
        
        # Treatment Actions
        elif action in [ActionType.APPLY_DISEASE_TREATMENT, ActionType.APPLY_PEST_CONTROL]:
            improvement = (prev_hive.disease_level - hive.disease_level) * 10
            score = 0.4 + min(0.6, improvement)
        
        # Emergency Actions
        elif action >= ActionType.EMERGENCY_RELOCATION:
            score = 0.9 if prev_hive.health_score < 30 else -0.2
            
        return np.clip(score, -0.5, 1.0)

    def _apply_action(self, action: int, hive: HiveState) -> None:
        """Apply the selected action to the hive"""
        action_type = ActionType(action)
        
        # Location Management Actions
        if action_type == ActionType.MOVE_TO_HIGH_NECTAR_ZONE:
            best_zone = max(self.nectar_zones, key=lambda z: z[2])
            hive.location = (best_zone[0] + random.uniform(-5, 5), 
                           best_zone[1] + random.uniform(-5, 5))
            hive.nectar_availability = min(1.0, best_zone[2] + random.uniform(-0.1, 0.1))
            
        elif action_type == ActionType.MOVE_TO_WATER_SOURCE:
            water_source = random.choice(self.water_sources)
            hive.location = (water_source[0] + random.uniform(-3, 3),
                           water_source[1] + random.uniform(-3, 3))
            hive.health_score = min(100, hive.health_score + random.uniform(2, 5))
            
        elif action_type == ActionType.RETURN_TO_BASE_LOCATION:
            hive.location = (self.base_location[0] + random.uniform(-10, 10),
                           self.base_location[1] + random.uniform(-10, 10))
            
        elif action_type == ActionType.MOVE_TO_PROTECTED_AREA:
            hive.location = (random.uniform(10, 30), random.uniform(10, 30))
            hive.predator_risk = max(0, hive.predator_risk - random.uniform(0.2, 0.4))
            hive.weather_risk = max(0, hive.weather_risk - random.uniform(0.1, 0.3))
        
        # Hive Maintenance Actions
        elif action_type == ActionType.SCHEDULE_IMMEDIATE_INSPECTION:
            hive.last_inspection = 0
            hive.health_score = min(100, hive.health_score + random.uniform(1, 3))
            
        elif action_type == ActionType.PROVIDE_SUGAR_FEEDING:
            hive.food_stores = min(100, hive.food_stores + random.uniform(5, 15))
            hive.bee_population = min(100000, hive.bee_population + random.randint(1000, 5000))
            
        elif action_type == ActionType.APPLY_DISEASE_TREATMENT:
            hive.disease_level = max(0, hive.disease_level - random.uniform(0.3, 0.7))
            hive.health_score = min(100, hive.health_score + random.uniform(5, 15))
            
        elif action_type == ActionType.APPLY_PEST_CONTROL:
            hive.pest_level = max(0, hive.pest_level - random.uniform(0.4, 0.8))
            hive.health_score = min(100, hive.health_score + random.uniform(3, 8))
        
        # Population Management Actions
        elif action_type == ActionType.RECOMMEND_HIVE_SPLITTING:
            if hive.bee_population > 50000:
                hive.bee_population = int(hive.bee_population * 0.6)
                hive.honey_production *= 0.7  # Temporary reduction
                
        elif action_type == ActionType.INTRODUCE_NEW_QUEEN:
            hive.queen_present = True
            hive.queen_age = random.randint(1, 3)
            hive.health_score = min(100, hive.health_score + random.uniform(10, 20))
            hive.productivity_trend = min(1, hive.productivity_trend + random.uniform(0.2, 0.5))
        
        # Environmental Adaptation Actions
        elif action_type == ActionType.INSTALL_SHADE_PROTECTION:
            if hive.temperature > 30:
                hive.temperature = max(25, hive.temperature - random.uniform(3, 8))
                hive.health_score = min(100, hive.health_score + random.uniform(2, 5))
                
        elif action_type == ActionType.IMPROVE_VENTILATION:
            hive.humidity = max(0.3, hive.humidity - random.uniform(0.1, 0.3))
            hive.health_score = min(100, hive.health_score + random.uniform(1, 4))
        
        # Monitoring Actions
        elif action_type == ActionType.DEPLOY_SOUND_MONITORING:
            hive.sound_activity = min(1.0, hive.sound_activity + random.uniform(0.1, 0.3))
            
        elif action_type == ActionType.MEASURE_HIVE_WEIGHT:
            # This would help estimate honey stores
            hive.honey_production += random.uniform(0.5, 2.0)
        
        # Resource Management Actions
        elif action_type == ActionType.HARVEST_HONEY:
            if hive.honey_production > 5:
                harvest_amount = min(hive.honey_production * 0.7, hive.food_stores * 0.3)
                self.episode_honey_production += harvest_amount
                hive.food_stores = max(5, hive.food_stores - harvest_amount)
                
        elif action_type == ActionType.LEAVE_HONEY_FOR_WINTER:
            # Ensure adequate winter stores
            if hive.season == 2:  # Autumn
                hive.food_stores = min(100, hive.food_stores + random.uniform(5, 15))
        
        # Emergency Actions (more dramatic effects)
        elif action_type == ActionType.EMERGENCY_RELOCATION:
            # Move to safest known location
            hive.location = (25, 25)  # Protected area
            hive.weather_risk = max(0, hive.weather_risk - 0.5)
            hive.predator_risk = max(0, hive.predator_risk - 0.4)
            
        elif action_type == ActionType.EMERGENCY_FEEDING:
            hive.food_stores = min(100, hive.food_stores + random.uniform(15, 30))
            hive.health_score = min(100, hive.health_score + random.uniform(5, 10))
            
        elif action_type == ActionType.EMERGENCY_TREATMENT:
            hive.disease_level = max(0, hive.disease_level - random.uniform(0.5, 0.9))
            hive.pest_level = max(0, hive.pest_level - random.uniform(0.4, 0.8))
            hive.health_score = min(100, hive.health_score + random.uniform(10, 25))
        
        # Update derived states
        self._update_hive_dynamics(hive)
    
    def _update_hive_dynamics(self, hive: HiveState) -> None:
        """More forgiving environment dynamics"""
        # Aging
        hive.hive_age += 1/30
        hive.queen_age += 1/30
        hive.last_inspection += 1
        
        # Season effects
        season = min(3, max(0, self.current_step // 91))
        seasonal_factor = [1.2, 1.0, 0.8, 0.6][season]
        
        # Health degradation (scaled to current health)
        if hive.last_inspection > 14:
            hive.health_score -= 0.05 * hive.last_inspection * (1 - hive.health_score/100)
        
        # Disease/Pest progression (diminishing returns)
        hive.disease_level = min(1, hive.disease_level * (1 + 0.03 * (1 - hive.disease_level)))
        hive.pest_level = min(1, hive.pest_level * (1 + 0.02 * (1 - hive.pest_level)))
        
        # Food consumption (seasonally adjusted)
        consumption = (hive.bee_population / 40000) * seasonal_factor
        hive.food_stores = max(0, hive.food_stores - consumption)
        
        # Queen effects
        if not hive.queen_present:
            hive.bee_population *= 0.98  # 2% decline
        
        # Productivity trend
        base_production = hive.honey_production
        hive.honey_production *= seasonal_factor * (0.9 + 0.2 * random.random())
        trend_change = (hive.honey_production - base_production) / max(1, base_production)
        hive.productivity_trend = 0.7 * hive.productivity_trend + 0.3 * trend_change
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step with new reward system"""
        self.selected_hive_id = self.current_step % self.num_hives
        target_hive = self.hives[self.selected_hive_id]
        prev_hive = dataclasses.replace(target_hive)  # Snapshot pre-state
        
        self._apply_action(action, target_hive)
        
        # Update all hives
        for hive in self.hives:
            self._update_hive_dynamics(hive)
        
        # Calculate reward
        reward = self._calculate_reward(action, target_hive, prev_hive)
        self.total_reward += reward
        
        # Termination checks
        terminated = self.current_step >= self.max_steps
        active_hives = sum(1 for hive in self.hives if hive.bee_population > 1000)
        if active_hives == 0:
            terminated = True
            reward -= 5  # Moderate penalty for total loss
            
        # Enhanced metrics
        info = {
            "total_reward": self.total_reward,
            "active_hives": active_hives,
            "reward_components": {
                "core": float(3 * (0.5*(target_hive.health_score/100) + 0.5*(target_hive.bee_population/80000))),
                "productivity": float(2.5 * (0.7*np.tanh(target_hive.honey_production - prev_hive.honey_production) + 
                                          0.3*target_hive.productivity_trend)),
                "action": float(2 * self._evaluate_action_effectiveness(ActionType(action), target_hive, prev_hive)),
                "risk": float(-1.5 * (target_hive.disease_level + target_hive.pest_level + 
                                    0.5*target_hive.predator_risk)),
                "sustainability": float((np.log1p(target_hive.food_stores)/5 + 
                                      (0.2 if target_hive.queen_present else 0)))
            },
            "health_delta": target_hive.health_score - prev_hive.health_score,
            "population_delta": target_hive.bee_population - prev_hive.bee_population
        }
        
        self.current_step += 1
        return self._get_observation(), reward, terminated, False, info
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset environment state
        self.current_step = 0
        self.total_reward = 0
        self.episode_honey_production = 0
        self.episode_hive_losses = 0
        self.selected_hive_id = 0
        
        # Reinitialize hives
        self.hives = [self._create_random_hive(i) for i in range(self.num_hives)]
        
        # Regenerate nectar zones
        self.nectar_zones = self._generate_nectar_zones()
        
        info = {
            "total_hives": self.num_hives,
            "world_size": self.world_size,
            "max_steps": self.max_steps
        }
        
        return self._get_observation(), info
    
    def render(self):
        """Render the environment - will be implemented in rendering.py"""
        if self.render_mode == "human":
            # This will be handled by the rendering module
            pass
        elif self.render_mode == "rgb_array":
            # Return RGB array representation
            return np.zeros((400, 400, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up rendering resources"""
        pass

# Test the environment
# Enhanced Test Script for BeehiveManagementEnv
if __name__ == "__main__":
    # Configuration
    TEST_CONFIG = {
        "num_hives": 3,
        "render_mode": "human",  # "human" or None
        "max_episodes": 3,
        "max_steps": 100,
        "strategic_action_freq": 5,  # Every N steps
        "pause_on_extreme": True,
        "log_level": "detailed"  # "minimal" or "detailed"
    }

    # Initialize
    env = BeehiveManagementEnv(
        num_hives=TEST_CONFIG["num_hives"],
        render_mode=TEST_CONFIG["render_mode"]
    )
    
    # Test statistics
    test_stats = {
        "episodes": [],
        "action_types": {action.name: 0 for action in ActionType},
        "reward_ranges": {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }
    }

    print("\n" + "="*40)
    print(f"Starting Beehive Environment Test")
    print(f"Mode: {TEST_CONFIG['render_mode'] or 'No rendering'}")
    print(f"Hives: {TEST_CONFIG['num_hives']} | Episodes: {TEST_CONFIG['max_episodes']}")
    print("="*40 + "\n")

    for episode in range(TEST_CONFIG["max_episodes"]):
        obs, info = env.reset()
        episode_stats = {
            "total_reward": 0,
            "hive_states": [],
            "actions_taken": [],
            "steps": 0,
            "termination": "completed"
        }

        print(f"\n=== Episode {episode + 1} ===")
        if TEST_CONFIG["log_level"] == "detailed":
            print("Initial Hive Status:")
            for i, hive in enumerate(env.hives):
                print(f"[Hive {i}] Bees: {hive.bee_population:>6} | Health: {hive.health_score:5.1f} | Food: {hive.food_stores:4.1f}kg")

        for step in range(TEST_CONFIG["max_steps"]):
            # Action selection strategy
            current_hive = env.hives[env.selected_hive_id]
            if step % TEST_CONFIG["strategic_action_freq"] == 0:
                # Strategic actions
                if current_hive.disease_level > 0.4:
                    action = ActionType.APPLY_DISEASE_TREATMENT
                elif current_hive.food_stores < 10:
                    action = ActionType.PROVIDE_SUGAR_FEEDING
                elif current_hive.health_score < 50:
                    action = ActionType.SCHEDULE_IMMEDIATE_INSPECTION
                else:
                    action = ActionType.MOVE_TO_HIGH_NECTAR_ZONE
            else:
                # Random exploration
                action = env.action_space.sample()
            
            test_stats["action_types"][ActionType(action).name] += 1

            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_stats["total_reward"] += reward
            episode_stats["steps"] = step + 1

            # Track reward distribution
            if reward > 5:
                test_stats["reward_ranges"]["positive"] += 1
            elif reward < -2:
                test_stats["reward_ranges"]["negative"] += 1
            else:
                test_stats["reward_ranges"]["neutral"] += 1

            # Log significant events
            if TEST_CONFIG["log_level"] == "detailed" and (abs(reward) > 3 or terminated):
                print(f"\nStep {step}: {ActionType(action).name}")
                print(f"Reward: {reward:7.2f} (Total: {episode_stats['total_reward']:7.2f})")
                print("-"*40)
                for k, v in info["reward_components"].items():
                    print(f"{k:>15}: {v:7.2f}")
                print(f"{'Health Δ':>15}: {info['health_delta']:7.1f}")
                print(f"{'Bees Δ':>15}: {info['population_delta']:7.0f}")
                
                if TEST_CONFIG["pause_on_extreme"] and abs(reward) > 5:
                    input("[Press Enter to continue]")

            if terminated or truncated:
                episode_stats["termination"] = "terminated" if terminated else "truncated"
                break

        # Record final hive states
        episode_stats["hive_states"] = [
            {
                "bees": hive.bee_population,
                "health": hive.health_score,
                "food": hive.food_stores,
                "status": "ALIVE" if hive.bee_population > 1000 else "DEAD"
            }
            for hive in env.hives
        ]
        test_stats["episodes"].append(episode_stats)

        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"Steps: {episode_stats['steps']} | Final Reward: {episode_stats['total_reward']:.2f}")
        print(f"Termination: {episode_stats['termination'].upper()}")
        print("Final Hive Status:")
        for i, state in enumerate(episode_stats["hive_states"]):
            print(f"[Hive {i}] {state['status']:<6} | Bees: {state['bees']:>6} | Health: {state['health']:5.1f} | Food: {state['food']:4.1f}kg")

        # Visual pause between episodes
        if TEST_CONFIG["render_mode"] == "human":
            env.render()
            time.sleep(2)  # Pause 2 seconds

    # Full test summary
    print("\n" + "="*40)
    print("TEST SUMMARY")
    print("="*40)
    
    # Reward distribution
    total_steps = sum(ep["steps"] for ep in test_stats["episodes"])
    print(f"\nReward Distribution ({total_steps} total steps):")
    print(f"Positive (>5): {test_stats['reward_ranges']['positive']:>4} ({test_stats['reward_ranges']['positive']/total_steps:.1%})")
    print(f"Neutral  (-2-5): {test_stats['reward_ranges']['neutral']:>4} ({test_stats['reward_ranges']['neutral']/total_steps:.1%})")
    print(f"Negative (<-2): {test_stats['reward_ranges']['negative']:>4} ({test_stats['reward_ranges']['negative']/total_steps:.1%})")

    # Action frequency
    print("\nMost Frequent Actions:")
    sorted_actions = sorted(test_stats["action_types"].items(), key=lambda x: x[1], reverse=True)[:5]
    for action, count in sorted_actions:
        print(f"{action:<30}: {count:>3} ({count/total_steps:.1%})")

    # Hive survival rate
    total_hives = TEST_CONFIG["num_hives"] * TEST_CONFIG["max_episodes"]
    dead_hives = sum(1 for ep in test_stats["episodes"] for hive in ep["hive_states"] if hive["status"] == "DEAD")
    print(f"\nHive Survival Rate: {(total_hives - dead_hives)/total_hives:.1%}")

    env.close()
    print("\nTest completed successfully!")
