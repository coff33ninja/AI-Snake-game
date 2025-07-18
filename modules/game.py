"""
Game Logic Module

This module contains the core game mechanics including:
- Game state management
- Game loop
- Collision detection
- Food generation
- Game mode logic
- Scoring system
"""

import random
import curses
import time
import logging

# Import from other modules
from .ai import SnakeGameAI
from .config import (DIFFICULTY_CONFIG, TURBO_DURATION, TURBO_COOLDOWN,
                     TURBO_SPEED_MULTIPLIER, MIN_EXPLORATION_RATE, EXPLORATION_RATE,
                     EXPLORATION_DECAY, REWARD_FOOD, REWARD_FOOD_P1, REWARD_COLLISION,
                     CURSES_TIMEOUT)


class GameState:
    """Manages the current state of the game"""

    def __init__(self, config, height, width):
        self.config = config
        self.height = height
        self.width_p = width // 2  # Width per player
        self.width = width

        # Game timing
        self.game_start_time = time.time()
        self.last_time = time.time()
        self.base_game_speed = DIFFICULTY_CONFIG[config['difficulty']]['speed']
        self.game_speed = self.base_game_speed

        # Game state flags
        self.game_won = False
        self.winner = None
        self.game_ended = False
        self.game_over = False  # Renamed from game_ended for clarity
        self.win_condition = None
        self.play_again = False

        # Player states
        self.p1_game_over = False
        self.p2_game_over = False
        self.p1_death_time = 0
        self.p2_death_time = 0

        # Scores
        self.p1_score = 0
        self.p2_score = 0

        # Initialize snakes
        self.p1_snake = [
            [height // 2, (self.width_p // 2)],
            [height // 2, (self.width_p // 2) - 1],
            [height // 2, (self.width_p // 2) - 2]
        ]
        self.p2_snake = [
            [height // 2, (self.width_p // 2)],
            [height // 2, (self.width_p // 2) - 1],
            [height // 2, (self.width_p // 2) - 2]
        ]

        # Directions
        self.p1_direction = curses.KEY_RIGHT
        self.p2_direction = curses.KEY_RIGHT

        # Game elements
        self.obstacles = []
        self.foods = []
        self.init_game_elements()

        # Turbo feature
        self.turbo_active = False
        self.turbo_start_time = 0
        self.turbo_cooldown_start = 0
        self.turbo_available = True

        # AI
        self.ai = None
        if config.get('p2_type') == 'ai' and not config.get('multiplayer_enabled', False):
            self.ai = SnakeGameAI(height, self.width_p)
            logging.info("Initialized AI for Player 2")

    def init_game_elements(self):
        """Initialize obstacles and food based on difficulty"""
        difficulty_config = DIFFICULTY_CONFIG[self.config['difficulty']]

        # Create obstacles (adjusted for screen size)
        self.obstacles = []
        if 'obstacles' in difficulty_config:
            for obs in difficulty_config['obstacles']:
                if obs[0] < self.height - 2 and obs[1] < self.width_p - 2:
                    self.obstacles.append(obs)

        # Create multiple foods
        self.foods = []
        if 'food_count' in difficulty_config:
            for _ in range(difficulty_config['food_count']):
                attempts = 0
                while attempts < 50:
                    new_food = [random.randint(4, self.height - 3), random.randint(2, self.width_p - 3)]
                    if (new_food not in self.p1_snake and new_food not in self.p2_snake and
                        new_food not in self.obstacles and new_food not in self.foods):
                        self.foods.append(new_food)
                        break
                    attempts += 1

    def update_turbo(self, current_time):
        """Update turbo state based on timing"""
        if self.turbo_active and current_time - self.turbo_start_time >= TURBO_DURATION:
            self.turbo_active = False
            self.turbo_cooldown_start = current_time
            self.turbo_available = False
            self.game_speed = self.base_game_speed
            logging.debug("Turbo mode ended")

        if not self.turbo_available and current_time - self.turbo_cooldown_start >= TURBO_COOLDOWN:
            self.turbo_available = True
            logging.debug("Turbo mode available again")

    def activate_turbo(self, current_time):
        """Activate turbo mode if available"""
        if not self.turbo_active and self.turbo_available:
            self.turbo_active = True
            self.turbo_start_time = current_time
            self.game_speed = self.base_game_speed / TURBO_SPEED_MULTIPLIER
            logging.info("Turbo mode activated")
            return True
        return False

    def check_collisions(self):
        """Check for collisions for both players"""
        p1_collision = False
        p2_collision = False

        # Check P1 collisions
        if not self.p1_game_over:
            p1_head = self.p1_snake[0]
            if (p1_head[0] <= 0 or p1_head[0] >= self.height - 1 or
                p1_head[1] <= 0 or p1_head[1] >= self.width_p - 2 or
                p1_head in self.p1_snake[1:] or p1_head in self.obstacles):
                p1_collision = True
                self.p1_game_over = True
                self.p1_death_time = time.time()
                self.win_condition = 'collision'
                logging.info(f"P1 collision detected! Head: {p1_head}")

        # Check P2 collisions
        if not self.p2_game_over:
            p2_head = self.p2_snake[0]
            if (p2_head[0] <= 0 or p2_head[0] >= self.height - 1 or
                p2_head[1] <= 0 or p2_head[1] >= self.width_p - 2 or
                p2_head in self.p2_snake[1:] or p2_head in self.obstacles):
                p2_collision = True
                self.p2_game_over = True
                self.p2_death_time = time.time()
                self.win_condition = 'collision'
                logging.info(f"P2 collision detected! Head: {p2_head}")

        return p1_collision, p2_collision

    def check_food_eaten(self):
        """Check if food was eaten by either player"""
        food_eaten = False
        reward = 0

        if len(self.foods) > 0:
            for i, food in enumerate(self.foods):
                if not self.p1_game_over and self.p1_snake[0] == food:
                    self.p1_score += 10
                    food_eaten = True
                    self.foods.pop(i)
                    reward = REWARD_FOOD_P1
                    logging.info(f"P1 ate food! New score: {self.p1_score}")

                    # Check win condition for First to Score mode
                    if self.config['game_mode'] == 2 and self.p1_score >= self.config['target_score']:
                        self.game_won = True
                        self.winner = "Player 1"
                        self.game_over = True
                        self.win_condition = 'target_score'
                    break
                elif not self.p2_game_over and self.p2_snake[0] == food:
                    self.p2_score += 10
                    food_eaten = True
                    self.foods.pop(i)
                    reward = REWARD_FOOD if self.ai else 0
                    logging.info(f"P2 ate food! New score: {self.p2_score}")

                    # Check win condition for First to Score mode
                    if self.config['game_mode'] == 2 and self.p2_score >= self.config['target_score']:
                        self.game_won = True
                        self.winner = "Player 2"
                        self.game_over = True
                        self.win_condition = 'target_score'
                    break

        return food_eaten, reward

    def generate_new_food(self):
        """Generate new food after one was eaten"""
        attempts = 0
        while attempts < 100:
            new_food = [random.randint(4, self.height - 3), random.randint(2, self.width_p - 3)]
            # Check collision with living snakes only
            collision_found = False
            if not self.p1_game_over and new_food in self.p1_snake:
                collision_found = True
            if not self.p2_game_over and new_food in self.p2_snake:
                collision_found = True
            if new_food in self.obstacles:
                collision_found = True
            if not collision_found:
                self.foods.append(new_food)
                logging.debug(f"Generated new food at {new_food}")
                break
            attempts += 1

        # If we can't find a spot, just place it randomly
        if attempts >= 100:
            new_food = [random.randint(4, self.height - 3), random.randint(2, self.width_p - 3)]
            self.foods.append(new_food)
            logging.warning(f"Could not find free spot for food after 100 attempts!")

    def update_snake_positions(self, p1_head, p2_head, food_eaten, p1_collision, p2_collision):
        """Update snake positions based on movement and collisions"""
        # Add new heads only if players are alive
        if not self.p1_game_over and p1_head is not None:
            self.p1_snake.insert(0, p1_head)
        if not self.p2_game_over and p2_head is not None:
            self.p2_snake.insert(0, p2_head)

        # Remove tails if no food eaten
        if not food_eaten:
            if not self.p1_game_over and not p1_collision:
                self.p1_snake.pop()
            if not self.p2_game_over and not p2_collision:
                self.p2_snake.pop()

    def check_game_end_conditions(self):
        """Check various game end conditions"""
        # Check if game won (First to Score)
        if self.game_won:
            self.game_ended = True
            logging.info(f"Game won by {self.winner}! Final scores - P1: {self.p1_score}, P2: {self.p2_score}")
            return True

        # Check time limit (Time Attack mode)
        if self.config['game_mode'] == 3:
            elapsed = time.time() - self.game_start_time
            if elapsed >= self.config['game_time_minutes'] * 60:
                self.game_ended = True
                self.game_over = True
                self.win_condition = 'time_limit'
                if self.p1_score > self.p2_score:
                    self.winner = "Player 1"
                elif self.p2_score > self.p1_score:
                    self.winner = "Player 2"
                else:
                    self.winner = "Tie"
                logging.info(f"Time limit reached! Winner: {self.winner}")
                return True

        # Check if both players are dead
        if self.p1_game_over and self.p2_game_over:
            self.game_ended = True
            self.game_over = True
            self.win_condition = 'collision'
            if self.p1_score > self.p2_score:
                self.winner = "Player 1"
            elif self.p2_score > self.p1_score:
                self.winner = "Player 2"
            else:
                self.winner = "Tie"
            logging.info(f"Both players died! Winner: {self.winner}")
            return True

        return False

    @property
    def game_duration(self):
        """Calculate game duration in seconds"""
        return time.time() - self.game_start_time

    def to_dict(self):
        """Serialize the game state to a dictionary for network transfer."""
        return {
            'p1_snake': self.p1_snake,
            'p2_snake': self.p2_snake,
            'foods': self.foods,
            'obstacles': self.obstacles,
            'p1_score': self.p1_score,
            'p2_score': self.p2_score,
            'p1_game_over': self.p1_game_over,
            'p2_game_over': self.p2_game_over,
            'game_won': self.game_won,
            'winner': self.winner,
            'game_ended': self.game_ended,
            'p1_direction': self.p1_direction,
            'p2_direction': self.p2_direction,
            'current_time': time.time(),
            'game_start_time': self.game_start_time,
            'config': self.config,
            'timestamp': time.time()  # Added for conflict resolution
        }

    def from_dict(self, state_dict):
        """Update the game state from a dictionary received over the network."""
        # Only update if the incoming state has a newer timestamp
        if 'timestamp' in state_dict and hasattr(self, 'last_update_time') and state_dict['timestamp'] < self.last_update_time:
            logging.debug("Ignoring outdated game state update")
            return
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_update_time = state_dict.get('timestamp', time.time())
        logging.debug("Applied game state update")


class GameEngine:
    """Main game engine that handles the game loop"""

    def __init__(self, game_state):
        self.game_state = game_state
        self.config = game_state.config
        self.height = game_state.height
        self.width = game_state.width

        # Will be set when run() is called with stdscr
        self.stdscr = None
        self.p1_win = None
        self.p2_win = None

        # Multiplayer client (if enabled)
        self.multiplayer_client = self.config.get('multiplayer_client', None)
        self.is_multiplayer = self.multiplayer_client is not None
        self.is_host = self.is_multiplayer and self.multiplayer_client.is_host
        self.opponent_joined = not self.is_host if self.is_multiplayer else True
        self.last_sync_time = 0

    def draw_separator(self):
        """Draw the vertical separator between player screens"""
        self.stdscr.clear()
        for i in range(self.height):
            self.stdscr.addch(i, self.width // 2, '|')
        self.stdscr.refresh()

    def handle_input(self):
        """Handle user input, adapting for multiplayer roles."""
        key = self.p1_win.getch()

        # Handle quit
        if key == ord('q') or key == ord('Q'):
            logging.info("Game quit by user")
            return False

        # Handle turbo
        if key == ord(' '):
            self.game_state.activate_turbo(time.time())

        # Determine which direction to update based on role
        current_direction = self.game_state.p1_direction if self.is_host else self.game_state.p2_direction
        new_direction = None

        if key in [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT]:
            if (key == curses.KEY_UP and current_direction != curses.KEY_DOWN) or \
               (key == curses.KEY_DOWN and current_direction != curses.KEY_UP) or \
               (key == curses.KEY_LEFT and current_direction != curses.KEY_RIGHT) or \
               (key == curses.KEY_RIGHT and current_direction != curses.KEY_LEFT):
                new_direction = key

        if new_direction:
            if self.is_multiplayer:
                self.multiplayer_client.send_message({"type": "player_move", "data": {"direction": new_direction, "player_id": self.multiplayer_client.client_id, "timestamp": time.time()}})
            if not self.is_multiplayer or self.is_host:
                self.game_state.p1_direction = new_direction
            else:
                self.game_state.p2_direction = new_direction
            logging.debug(f"Updated direction: {new_direction}")

        return True

    def calculate_new_positions(self):
        """Calculate new head positions for both players"""
        p1_head = None
        p2_head = None

        # Calculate P1 new position
        if not self.game_state.p1_game_over:
            p1_head = self.game_state.p1_snake[0].copy()
            if self.game_state.p1_direction == curses.KEY_UP:
                p1_head[0] -= 1
            elif self.game_state.p1_direction == curses.KEY_DOWN:
                p1_head[0] += 1
            elif self.game_state.p1_direction == curses.KEY_LEFT:
                p1_head[1] -= 1
            elif self.game_state.p1_direction == curses.KEY_RIGHT:
                p1_head[1] += 1

        # Calculate P2 new position
        if not self.game_state.p2_game_over:
            p2_head = self.game_state.p2_snake[0].copy()
            if self.is_multiplayer and self.config.get('p2_type') == 'ai':
                # Server-hosted AI
                p2_state = self.game_state.ai.get_state(self.game_state.p2_snake, self.game_state.foods[0] if self.game_state.foods else [0, 0], self.game_state.p2_direction, self.game_state.obstacles)
                p2_action = self.multiplayer_client.get_ai_move(p2_state)
                if p2_action is not None:
                    self.game_state.p2_direction = p2_action
            elif self.game_state.ai:  # If P2 is local AI
                closest_food = self.game_state.foods[0] if self.game_state.foods else [0, 0]
                if len(self.game_state.foods) > 1:
                    p2_head_current = self.game_state.p2_snake[0]
                    distances = [abs(p2_head_current[0] - f[0]) + abs(p2_head_current[1] - f[1]) for f in self.game_state.foods]
                    closest_food = self.game_state.foods[distances.index(min(distances))]
                p2_state = self.game_state.ai.get_state(self.game_state.p2_snake, closest_food, self.game_state.p2_direction, self.game_state.obstacles)
                p2_action = self.game_state.ai.get_action(p2_state, DIFFICULTY_CONFIG[self.config['difficulty']]['ai_exploration'])
                if (p2_action == curses.KEY_UP and self.game_state.p2_direction != curses.KEY_DOWN) or \
                   (p2_action == curses.KEY_DOWN and self.game_state.p2_direction != curses.KEY_UP) or \
                   (p2_action == curses.KEY_LEFT and self.game_state.p2_direction != curses.KEY_RIGHT) or \
                   (p2_action == curses.KEY_RIGHT and self.game_state.p2_direction != curses.KEY_LEFT):
                    self.game_state.p2_direction = p2_action
            if self.game_state.p2_direction == curses.KEY_UP:
                p2_head[0] -= 1
            elif self.game_state.p2_direction == curses.KEY_DOWN:
                p2_head[0] += 1
            elif self.game_state.p2_direction == curses.KEY_LEFT:
                p2_head[1] -= 1
            elif self.game_state.p2_direction == curses.KEY_RIGHT:
                p2_head[1] += 1

        return p1_head, p2_head

    def update_ai_learning(self, reward):
        """Update AI learning based on game state"""
        if self.is_multiplayer and self.config.get('p2_type') == 'ai':
            # Server-hosted AI
            p2_state = self.game_state.ai.get_state(self.game_state.p2_snake, self.game_state.foods[0] if self.game_state.foods else [0, 0], self.game_state.p2_direction, self.game_state.obstacles)
            p2_next_state = self.game_state.ai.get_state(self.game_state.p2_snake, self.game_state.foods[0] if self.game_state.foods else [0, 0], self.game_state.p2_direction, self.game_state.obstacles)
            self.multiplayer_client.send_ai_training_data(p2_state, self.game_state.p2_direction, reward, p2_next_state, self.game_state.p2_game_over)
        elif self.game_state.ai and not self.game_state.p2_game_over:
            closest_food = self.game_state.foods[0] if self.game_state.foods else [0, 0]
            if len(self.game_state.foods) > 1:
                p2_head_current = self.game_state.p2_snake[0]
                distances = [abs(p2_head_current[0] - f[0]) + abs(p2_head_current[1] - f[1]) for f in self.game_state.foods]
                closest_food = self.game_state.foods[distances.index(min(distances))]
            p2_state = self.game_state.ai.get_state(self.game_state.p2_snake, closest_food, self.game_state.p2_direction, self.game_state.obstacles)
            p2_action = self.game_state.ai.get_action(p2_state, DIFFICULTY_CONFIG[self.config['difficulty']]['ai_exploration'])
            p2_next_state = self.game_state.ai.get_state(self.game_state.p2_snake, closest_food, self.game_state.p2_direction, self.game_state.obstacles)
            self.game_state.ai.update_q_table(p2_state, p2_action, reward, p2_next_state)
            logging.debug(f"Updated AI Q-table with reward: {reward}")

    def draw_game_elements(self):
        """Draw all game elements on the screen"""
        current_time = time.time()

        # Clear windows
        self.p1_win.clear()
        self.p2_win.clear()

        # Draw window borders
        self.p1_win.box()
        self.p2_win.box()

        # Display scores and game info
        if not self.game_state.p1_game_over:
            self.p1_win.addstr(1, 2, f"P1 Score: {self.game_state.p1_score}")
            self.p1_win.addstr(2, 2, f"Difficulty: {DIFFICULTY_CONFIG[self.config['difficulty']]['name']}")
            self.p1_win.addstr(3, 2, f"Arrow keys, 'q' quit")

            # Display game mode specific info
            if self.config['game_mode'] == 2:  # First to Score
                self.p1_win.addstr(4, 2, f"Target: {self.config['target_score']}")
            elif self.config['game_mode'] == 3:  # Time Attack
                elapsed = time.time() - self.game_state.game_start_time
                remaining = max(0, self.config['game_time_minutes'] * 60 - elapsed)
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                self.p1_win.addstr(4, 2, f"Time: {minutes:02d}:{seconds:02d}")

            # Turbo status display
            turbo_line = 5 if self.config['game_mode'] in [2, 3] else 4
            if self.game_state.turbo_active:
                turbo_remaining = TURBO_DURATION - (current_time - self.game_state.turbo_start_time)
                self.p1_win.addstr(turbo_line, 2, f"TURBO: {turbo_remaining:.1f}s")
            elif not self.game_state.turbo_available:
                cooldown_remaining = TURBO_COOLDOWN - (current_time - self.game_state.turbo_cooldown_start)
                self.p1_win.addstr(turbo_line, 2, f"Cooldown: {cooldown_remaining:.1f}s")
            else:
                self.p1_win.addstr(turbo_line, 2, f"SPACE for turbo")

        if not self.game_state.p2_game_over:
            self.p2_win.addstr(1, 2, f"P2 (AI) Score: {self.game_state.p2_score}")
            self.p2_win.addstr(2, 2, f"Learning...")

        # Draw food
        if not self.game_state.p1_game_over or not self.game_state.p2_game_over:
            for food in self.game_state.foods:
                try:
                    if not self.game_state.p1_game_over:
                        self.p1_win.addch(food[0], food[1], '*')
                    if not self.game_state.p2_game_over:
                        self.p2_win.addch(food[0], food[1], '*')
                except curses.error:
                    pass

        # Draw obstacles
        for obstacle in self.game_state.obstacles:
            try:
                if not self.game_state.p1_game_over:
                    self.p1_win.addch(obstacle[0], obstacle[1], 'X')
                if not self.game_state.p2_game_over:
                    self.p2_win.addch(obstacle[0], obstacle[1], 'X')
            except curses.error:
                pass

        # Draw snakes or game over screens
        if not self.game_state.p1_game_over:
            for i, segment in enumerate(self.game_state.p1_snake):
                try:
                    self.p1_win.addch(segment[0], segment[1], 'O' if i == 0 else '#')
                except curses.error:
                    pass
        else:
            # Draw P1 game over screen
            self.draw_game_over_screen(self.p1_win, 1, self.game_state.p1_score,
                                     current_time - self.game_state.p1_death_time,
                                     not self.game_state.p2_game_over)

        if not self.game_state.p2_game_over:
            for i, segment in enumerate(self.game_state.p2_snake):
                try:
                    self.p2_win.addch(segment[0], segment[1], 'O' if i == 0 else '#')
                except curses.error:
                    pass
        else:
            # Draw P2 game over screen
            self.draw_game_over_screen(self.p2_win, 2, self.game_state.p2_score,
                                     current_time - self.game_state.p2_death_time,
                                     not self.game_state.p1_game_over)

        # Refresh screens
        self.p1_win.refresh()
        self.p2_win.refresh()

    def draw_game_over_screen(self, win, player_num, score, death_time, other_alive):
        """Draw game over screen for a player"""
        try:
            win.addstr(self.height // 2 - 3, 2, "GAME OVER")
            win.addstr(self.height // 2 - 2, 2, f"Player {player_num} Died!")
            win.addstr(self.height // 2 - 1, 2, f"Final Score: {score}")
            win.addstr(self.height // 2, 2, f"Died: {death_time:.1f}s ago")
            if other_alive:
                win.addstr(self.height // 2 + 2, 2, f"Player {3-player_num} continues...")
                win.addstr(self.height // 2 + 3, 2, "Press 'q' to quit")
            else:
                win.addstr(self.height // 2 + 2, 2, "Both players died!")
        except curses.error:
            pass

    def show_final_screen(self):
        """Show the final game over screen"""
        try:
            self.stdscr.clear()

            if self.game_state.game_won:
                self.stdscr.addstr(self.height // 2 - 3, self.width // 2 - 8, "GAME WON!")
                self.stdscr.addstr(self.height // 2 - 2, self.width // 2 - 10, f"Winner: {self.game_state.winner}")
            elif self.config['game_mode'] == 3:
                self.stdscr.addstr(self.height // 2 - 3, self.width // 2 - 8, "TIME'S UP!")
                self.stdscr.addstr(self.height // 2 - 2, self.width // 2 - 10, f"Winner: {self.game_state.winner}")
            else:
                self.stdscr.addstr(self.height // 2 - 3, self.width // 2 - 10, "BOTH PLAYERS DIED!")
                self.stdscr.addstr(self.height // 2 - 2, self.width // 2 - 10, f"Winner: {self.game_state.winner}")

            self.stdscr.addstr(self.height // 2, self.width // 2 - 15, f"Player 1 Final Score: {self.game_state.p1_score}")
            self.stdscr.addstr(self.height // 2 + 1, self.width // 2 - 15, f"Player 2 Final Score: {self.game_state.p2_score}")

            if self.config['game_mode'] == 2:
                self.stdscr.addstr(self.height // 2 + 3, self.width // 2 - 12, f"Target Score: {self.config['target_score']}")
            elif self.config['game_mode'] == 3:
                self.stdscr.addstr(self.height // 2 + 3, self.width // 2 - 12, f"Game Time: {self.config['game_time_minutes']} minutes")

            self.stdscr.addstr(self.height // 2 + 5, self.width // 2 - 12, "Press 'P' to Play Again or 'Q' to Quit.")
            self.stdscr.refresh()
            while True:
                key = self.stdscr.getch()
                if key == ord('p') or key == ord('P'):
                    self.game_state.play_again = True
                    break
                elif key == ord('q') or key == ord('Q'):
                    self.game_state.play_again = False
                    break
        except curses.error:
            pass

    def sync_multiplayer(self):
        """Sync game state with multiplayer server if enabled"""
        if self.is_multiplayer and self.is_host:
            game_state_dict = self.game_state.to_dict()
            self.multiplayer_client.send_message({
                "type": "game_state_update",
                "data": game_state_dict
            })
            logging.debug("Sent game state update to server")

    def run(self, stdscr):
        """Main game loop"""
        global EXPLORATION_RATE

        # Initialize curses interface
        self.stdscr = stdscr
        self.stdscr.nodelay(True)

        # Create split-screen windows
        self.p1_win = curses.newwin(self.height, self.width // 2 - 1, 0, 0)
        self.p2_win = curses.newwin(self.height, self.width // 2 - 1, 0, self.width // 2 + 1)
        self.p1_win.keypad(True)
        self.p1_win.timeout(CURSES_TIMEOUT)
        self.p2_win.timeout(CURSES_TIMEOUT)

        # Draw screen separator
        self.draw_separator()

        while True:
            # Control game speed
            current_time = time.time()

            # Update turbo state
            self.game_state.update_turbo(current_time)

            if current_time - self.game_state.last_time < self.game_state.game_speed:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue
            self.game_state.last_time = current_time

            # Handle multiplayer messages
            if self.multiplayer_client:
                message = self.multiplayer_client.get_message()
                if message:
                    if message['type'] == 'game_state_update':
                        self.game_state.from_dict(message['data'])
                        logging.debug("Applied game state update from server")
                    elif message['type'] == 'player_move':
                        if not self.is_host:
                            self.game_state.p2_direction = message['data']['direction']
                        else:
                            self.game_state.p1_direction = message['data']['direction']
                        logging.debug(f"Received player move: {message['data']['direction']}")
                    elif message['type'] == 'player_joined':
                        self.opponent_joined = True
                        logging.info("Opponent joined the game")
                    elif message['type'] == 'session_ended':
                        self.game_state.game_ended = True
                        self.game_state.game_over = True
                        self.game_state.winner = "Session Ended"
                        logging.info("Session ended by server")
                    elif message['type'] == 'player_disconnected':
                        self.game_state.game_ended = True
                        self.game_state.game_over = True
                        self.game_state.winner = "Opponent Disconnected"
                        logging.info("Opponent disconnected")

            # Skip game logic until opponent joins
            if self.is_multiplayer and not self.opponent_joined:
                time.sleep(0.1)
                continue

            # Sync with multiplayer server
            self.sync_multiplayer()

            # Handle input
            if not self.handle_input():
                break  # User quit

            # Calculate new positions
            p1_head, p2_head = self.calculate_new_positions()

            # Check collisions
            p1_collision, p2_collision = self.game_state.check_collisions()

            # Handle AI reward for collisions
            reward = REWARD_COLLISION if p2_collision else 0

            # Check if food was eaten
            food_eaten, food_reward = self.game_state.check_food_eaten()
            reward += food_reward

            # Update snake positions
            self.game_state.update_snake_positions(p1_head, p2_head, food_eaten, p1_collision, p2_collision)

            # Generate new food if needed
            if food_eaten:
                self.game_state.generate_new_food()

            # Update AI learning
            self.update_ai_learning(reward)

            # Draw everything
            self.draw_game_elements()

            # Check game end conditions
            if self.game_state.check_game_end_conditions():
                self.show_final_screen()
                if self.game_state.ai:
                    self.game_state.ai.save_q_table()
                break

            # Update exploration rate
            if 'ai_exploration' in DIFFICULTY_CONFIG[self.config['difficulty']]:
                EXPLORATION_RATE = max(MIN_EXPLORATION_RATE, EXPLORATION_RATE * EXPLORATION_DECAY)

        if self.multiplayer_client:
            self.multiplayer_client.disconnect()
        return self.game_state
