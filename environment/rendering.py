import pygame
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional
from enum import IntEnum
import os
import sys
from custom_env import BeehiveManagementEnv, HiveState, ActionType
import colorsys

# Initialize Pygame (main.py handles this, but keep for standalone testing)
pygame.init()

class Colors:
    """Color constants for visualization"""
    # Environment colors
    BACKGROUND = (34, 49, 63)  # Dark blue-gray
    GRASS = (76, 153, 76)
    NECTAR_ZONE = (255, 255, 102)  # Light yellow
    WATER = (102, 178, 255)  # Light blue
    PROTECTED_AREA = (102, 255, 102)  # Light green

    # Hive colors based on health
    HIVE_EXCELLENT = (0, 255, 0)      # Bright green
    HIVE_GOOD = (127, 255, 0)         # Yellow-green
    HIVE_AVERAGE = (255, 255, 0)      # Yellow
    HIVE_POOR = (255, 127, 0)         # Orange
    HIVE_CRITICAL = (255, 0, 0)       # Red
    HIVE_DEAD = (128, 128, 128)       # Gray

    # UI colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    PANEL_BG = (45, 62, 80)
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (200, 200, 200)
    BUTTON_NORMAL = (70, 130, 180)
    BUTTON_HOVER = (100, 149, 237)
    ERROR_RED = (255, 50, 50)  # For debug overlay

    # Action visualization
    ACTION_MOVE = (255, 165, 0)       # Orange
    ACTION_MAINTAIN = (0, 191, 255)   # Deep sky blue
    ACTION_EMERGENCY = (255, 20, 147) # Deep pink
    ACTION_MONITOR = (138, 43, 226)   # Blue violet

class BeehiveRenderer:
    """Advanced visualization system for beehive management environment"""

    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.world_size = 100  # Environment coordinate system

        # Initialize display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Beehive Management RL Environment")

        # Fonts
        self.font_large = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 14)

        # Layout dimensions
        self.world_panel_width = 700
        self.info_panel_width = width - self.world_panel_width
        self.top_panel_height = 60

        # Animation and effects
        self.bee_particles = []
        self.action_effects = []
        self.time_step = 0

        # Sound visualization
        self.sound_waves = {}

        # Performance tracking
        self.frame_count = 0
        self.clock = pygame.time.Clock()

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_x = int((x / self.world_size) * (self.world_panel_width - 40) + 20)
        screen_y = int((y / self.world_size) * (self.height - self.top_panel_height - 40) +
                      self.top_panel_height + 20)
        return screen_x, screen_y

    def get_hive_color(self, hive: HiveState) -> Tuple[int, int, int]:
        """Get hive color based on health and status"""
        population = max(0, hive.bee_population) if hive.bee_population is not None else 0
        if population < 1000:
            return Colors.HIVE_DEAD
        elif hive.health_score >= 80:
            return Colors.HIVE_EXCELLENT
        elif hive.health_score >= 60:
            return Colors.HIVE_GOOD
        elif hive.health_score >= 40:
            return Colors.HIVE_AVERAGE
        elif hive.health_score >= 20:
            return Colors.HIVE_POOR
        else:
            return Colors.HIVE_CRITICAL

    def draw_background_environment(self, env: BeehiveManagementEnv):
        """Draw the environmental background with zones and features"""
        world_rect = pygame.Rect(0, self.top_panel_height, self.world_panel_width,
                                self.height - self.top_panel_height)
        pygame.draw.rect(self.screen, Colors.GRASS, world_rect)

        for zone_x, zone_y, richness in env.nectar_zones:
            screen_x, screen_y = self.world_to_screen(zone_x, zone_y)
            radius = int(20 * richness) + 10

            for i in range(radius, 0, -2):
                alpha = int(100 * richness * (i / radius))
                color = (*Colors.NECTAR_ZONE, alpha)
                circle_surf = pygame.Surface((i*2, i*2), pygame.SRCALPHA)
                pygame.draw.circle(circle_surf, color, (i, i), i)
                self.screen.blit(circle_surf, (screen_x - i, screen_y - i))

        for water_x, water_y in env.water_sources:
            screen_x, screen_y = self.world_to_screen(water_x, water_y)
            wave_offset = math.sin(self.time_step * 0.1) * 3
            pygame.draw.circle(self.screen, Colors.WATER,
                             (screen_x, screen_y), 15 + int(wave_offset))
            pygame.draw.circle(self.screen, (200, 230, 255),
                             (screen_x - 3, screen_y - 3), 8)

        protected_areas = [(15, 15), (85, 15), (15, 85), (85, 85)]
        for px, py in protected_areas:
            screen_x, screen_y = self.world_to_screen(px, py)
            pygame.draw.rect(self.screen, Colors.PROTECTED_AREA,
                           (screen_x - 25, screen_y - 25, 50, 50), 3)

    def draw_hive(self, hive: HiveState, selected: bool = False):
        """Draw individual hive with detailed visualization"""
        screen_x, screen_y = self.world_to_screen(hive.location[0], hive.location[1])
        hive_color = self.get_hive_color(hive)

        population = max(0, hive.bee_population) if hive.bee_population is not None else 0
        base_size = 20 + int(population / 5000)
        base_size = min(base_size, 40)

        if selected:
            pulse = int(10 * (1 + math.sin(self.time_step * 0.2)))
            pygame.draw.circle(self.screen, Colors.WHITE,
                             (screen_x, screen_y), base_size + pulse, 3)

        pygame.draw.circle(self.screen, hive_color, (screen_x, screen_y), base_size)
        pygame.draw.circle(self.screen, Colors.BLACK, (screen_x, screen_y), base_size, 2)

        if hive.queen_present:
            crown_points = [
                (screen_x - 6, screen_y - base_size + 5),
                (screen_x - 3, screen_y - base_size + 2),
                (screen_x, screen_y - base_size),
                (screen_x + 3, screen_y - base_size + 2),
                (screen_x + 6, screen_y - base_size + 5)
            ]
            pygame.draw.polygon(self.screen, (255, 215, 0), crown_points)

        bar_width = base_size * 2
        bar_height = 4
        bar_x = screen_x - bar_width // 2
        bar_y = screen_y + base_size + 5

        pygame.draw.rect(self.screen, Colors.BLACK,
                        (int(bar_x), int(bar_y), int(bar_width), int(bar_height)))

        health_width = int((hive.health_score / 100) * bar_width)
        health_color = hive_color
        if health_width > 0:  # Only draw if width is positive
            pygame.draw.rect(self.screen, health_color,
                            (int(bar_x), int(bar_y), int(health_width), int(bar_height)))

        if hive.hive_id in self.sound_waves:
            self.draw_sound_waves(screen_x, screen_y, hive.sound_activity)

        if hive.disease_level > 0.3:
            pygame.draw.circle(self.screen, (255, 0, 0),
                             (screen_x + base_size - 5, screen_y - base_size + 5), 3)

        if hive.pest_level > 0.3:
            pygame.draw.circle(self.screen, (255, 165, 0),
                             (screen_x - base_size + 5, screen_y - base_size + 5), 3)

        id_text = self.font_small.render(f"H{hive.hive_id}", True, Colors.WHITE)
        self.screen.blit(id_text, (screen_x - 8, screen_y + base_size + 15))

    def draw_sound_waves(self, x: int, y: int, activity: float):
        """Draw animated sound waves around active hives"""
        wave_count = int(activity * 3) + 1
        for i in range(wave_count):
            wave_radius = 30 + (i * 15) + (self.time_step * 2) % 30
            alpha = int(255 * activity * (1 - i / wave_count))

            if alpha > 20:
                wave_surf = pygame.Surface((wave_radius * 2, wave_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(wave_surf, (*Colors.BUTTON_NORMAL, alpha),
                                 (wave_radius, wave_radius), wave_radius, 2)
                self.screen.blit(wave_surf, (x - wave_radius, y - wave_radius))

    def draw_bee_particles(self, hives: List[HiveState]):
        """Draw animated bee particles flying between hives and nectar sources"""
        for particle in self.bee_particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1

            if particle['life'] <= 0:
                self.bee_particles.remove(particle)
            else:
                screen_x, screen_y = self.world_to_screen(particle['x'], particle['y'])
                size = max(1, int(particle['life'] / 20))
                pygame.draw.circle(self.screen, (255, 255, 0), (screen_x, screen_y), size)

        for hive in hives:
            population = max(0, hive.bee_population) if hive.bee_population is not None else 0
            if population > 5000 and random.random() < hive.sound_activity * 0.1:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0.5, 2.0)

                particle = {
                    'x': hive.location[0],
                    'y': hive.location[1],
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': random.randint(30, 60)
                }
                self.bee_particles.append(particle)

    def draw_action_effect(self, action: int, hive: HiveState):
        """Draw visual effect for the current action"""
        screen_x, screen_y = self.world_to_screen(hive.location[0], hive.location[1])

        if action <= 9:
            color = Colors.ACTION_MOVE
            effect_type = "MOVE"
        elif action <= 29:
            color = Colors.ACTION_MAINTAIN
            effect_type = "MAINTAIN"
        elif action <= 49:
            color = Colors.ACTION_MONITOR
            effect_type = "MONITOR"
        elif action >= 60 and action <= 69:
            color = Colors.ACTION_EMERGENCY
            effect_type = "EMERGENCY"
        else:
            color = Colors.BUTTON_NORMAL
            effect_type = "OTHER"

        effect = {
            'x': screen_x,
            'y': screen_y,
            'color': color,
            'type': effect_type,
            'life': 60,
            'action_name': ActionType(action).name
        }
        self.action_effects.append(effect)

    def draw_action_effects(self):
        """Draw and update action effect animations"""
        for effect in self.action_effects[:]:
            effect['life'] -= 1

            if effect['life'] <= 0:
                self.action_effects.remove(effect)
                continue

            alpha = int(255 * (effect['life'] / 60))
            radius = 60 - effect['life']

            if alpha > 10:
                effect_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(effect_surf, (*effect['color'], alpha),
                                 (radius, radius), radius, 3)
                self.screen.blit(effect_surf,
                               (effect['x'] - radius, effect['y'] - radius))

            if effect['life'] > 30:
                text = self.font_small.render(effect['type'], True, effect['color'])
                self.screen.blit(text, (effect['x'] - 20, effect['y'] - 50))

    def draw_info_panel(self, env: BeehiveManagementEnv, last_action: Optional[int] = None,
                       last_reward: float = 0):
        """Draw information panel with environment stats"""
        panel_x = self.world_panel_width
        panel_rect = pygame.Rect(panel_x, 0, self.info_panel_width, self.height)
        pygame.draw.rect(self.screen, Colors.PANEL_BG, panel_rect)

        y_offset = 20

        title = self.font_large.render("Beehive Management", True, Colors.TEXT_PRIMARY)
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 40

        season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
        season_index = min(3, max(0, env.current_step // 91))

        stats = [
            f"Step: {env.current_step}/{env.max_steps}",
            f"Season: {season_names[season_index]}",
            f"Active Hives: {sum(1 for h in env.hives if h.bee_population > 1000)}",
            f"Total Reward: {env.total_reward:.1f}",
            f"Last Reward: {last_reward:.2f}"
        ]

        for stat in stats:
            text = self.font_medium.render(stat, True, Colors.TEXT_PRIMARY)
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 25

        y_offset += 20

        if last_action is not None:
            action_title = self.font_medium.render("Last Action:", True, Colors.TEXT_PRIMARY)
            self.screen.blit(action_title, (panel_x + 10, y_offset))
            y_offset += 20

            action_name = ActionType(last_action).name.replace('_', ' ')
            words = action_name.split()
            lines = []
            current_line = []

            for word in words:
                if len(' '.join(current_line + [word])) > 20:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
                else:
                    current_line.append(word)

            if current_line:
                lines.append(' '.join(current_line))

            for line in lines:
                text = self.font_small.render(line, True, Colors.TEXT_SECONDARY)
                self.screen.blit(text, (panel_x + 15, y_offset))
                y_offset += 18

        y_offset += 20

        hive_title = self.font_medium.render("Hive Details:", True, Colors.TEXT_PRIMARY)
        self.screen.blit(hive_title, (panel_x + 10, y_offset))
        y_offset += 25

        for i, hive in enumerate(env.hives):
            if y_offset > self.height - 100:
                break

            hive_color = self.get_hive_color(hive)

            header = f"Hive {hive.hive_id}:"
            if i == env.selected_hive_id:
                header += " [SELECTED]"

            text = self.font_small.render(header, True, hive_color)
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 18

            population = max(0, hive.bee_population) if hive.bee_population is not None else 0
            hive_stats = [
                f"  Health: {hive.health_score:.0f}%",
                f"  Population: {population:,}",
                f"  Honey: {hive.honey_production:.1f}kg",
                f"  Queen: {'Yes' if hive.queen_present else 'No'}",
                f"  Food: {hive.food_stores:.1f}kg"
            ]

            for stat in hive_stats:
                text = self.font_small.render(stat, True, Colors.TEXT_SECONDARY)
                self.screen.blit(text, (panel_x + 10, y_offset))
                y_offset += 15

            y_offset += 10

    def draw_top_panel(self, env: BeehiveManagementEnv):
        """Draw top status bar"""
        panel_rect = pygame.Rect(0, 0, self.width, self.top_panel_height)
        pygame.draw.rect(self.screen, Colors.PANEL_BG, panel_rect)

        base_env = getattr(env, "env", env)
        progress = base_env.current_step / base_env.max_steps
        progress_width = int((self.world_panel_width - 40) * progress)

        pygame.draw.rect(self.screen, Colors.BLACK, (20, 20, self.world_panel_width - 40, 20))
        pygame.draw.rect(self.screen, Colors.BUTTON_NORMAL, (20, 20, progress_width, 20))

        progress_text = f"Day {base_env.current_step} / {base_env.max_steps}"
        text = self.font_medium.render(progress_text, True, Colors.TEXT_PRIMARY)
        self.screen.blit(text, (25, 22))

        season_names = ["🌸 Spring", "☀️ Summer", "🍂 Autumn", "❄️ Winter"]
        season_index = min(3, max(0, base_env.current_step // 91))
        season_text = self.font_medium.render(season_names[season_index], True, Colors.TEXT_PRIMARY)
        self.screen.blit(season_text, (self.world_panel_width - 150, 22))

    def draw_debug_overlay(self, messages: List[str]):
        """Draw debug messages on-screen"""
        y_offset = self.top_panel_height + 10
        for msg in messages:
            text = self.font_medium.render(msg, True, Colors.ERROR_RED)
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

    def render(self, env: BeehiveManagementEnv, last_action: Optional[int] = None,
               last_reward: float = 0) -> np.ndarray:
        """Main render function"""
        self.screen.fill(Colors.BACKGROUND)

        # Validate environment state
        debug_messages = []
        hives = getattr(env, 'hives', [])
        selected_hive_id = getattr(env, 'selected_hive_id', -1)
        if not hives:
            debug_messages.append("Error: No hives in environment")
            print("Warning: No hives in environment")
        if selected_hive_id < 0 or selected_hive_id >= len(hives):
            debug_messages.append(f"Error: Invalid selected_hive_id: {selected_hive_id}")
            print(f"Warning: Invalid selected_hive_id: {selected_hive_id}, Hives: {len(hives)}")

        self.draw_top_panel(env)
        self.draw_background_environment(env)

        for i, hive in enumerate(hives):
            selected = (i == selected_hive_id)
            self.draw_hive(hive, selected)

        self.draw_bee_particles(hives)
        self.draw_action_effects()

        if last_action is not None and selected_hive_id < len(hives):
            self.draw_action_effect(last_action, hives[selected_hive_id])

        self.draw_info_panel(env, last_action, last_reward)

        # Draw debug overlay if there are issues
        if debug_messages:
            self.draw_debug_overlay(debug_messages)

        pygame.display.flip()
        self.clock.tick(30)

        self.time_step += 1

        frame = pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))
        if np.mean(frame) < 10:  # Account for dark background (34, 49, 63)
            print(f"Warning: Rendered frame is nearly black (mean: {np.mean(frame):.2f})")
        return frame

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.display.quit()
            self.screen = None

if __name__ == "__main__":
    env = BeehiveManagementEnv(num_hives=4, render_mode="human")
    renderer = BeehiveRenderer()

    obs, info = env.reset()

    print("Testing renderer with random actions...")
    print("Press ESC to quit, SPACE to pause, or wait for auto-quit after 100 steps")

    running = True
    paused = False
    last_action = None
    last_reward = 0
    max_test_steps = 100
    test_step_count = 0

    while running and test_step_count < max_test_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")

        if not paused:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            last_action = action
            last_reward = reward
            test_step_count += 1

            print(f"Step {test_step_count}/{max_test_steps}: Action={ActionType(action).name[:20]}..., "
                  f"Reward={reward:.2f}, Active Hives={info['active_hives']}")

            if terminated or truncated:
                print("Episode finished! Resetting...")
                obs, info = env.reset()
                last_action = None
                last_reward = 0

        renderer.render(env, last_action, last_reward)

    print(f"Test completed after {test_step_count} steps!")
    renderer.close()
    env.close()