"""
AI module for Q-learning snake AI
"""

import random
import pickle
import os
import curses
import logging
from .config import LEARNING_RATE, DISCOUNT_FACTOR


class SnakeGameAI:

    def __init__(self, height, width):
        self.height = height
        self.width = width // 2  # Split screen
        self.q_table = {}
        self.actions = [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT]
        self.load_q_table()
        logging.debug("Initialized SnakeGameAI")

    def get_state(self, snake, food, direction, obstacles):
        """
        Get a discretized, binary state representation for Q-learning.
        This state is a 12-element tuple of binary flags.
        """
        head = snake[0]

        # Points to check for danger
        point_l = [head[0], head[1] - 1]
        point_r = [head[0], head[1] + 1]
        point_u = [head[0] - 1, head[1]]
        point_d = [head[0] + 1, head[1]]

        # Current direction (one-hot encoded)
        dir_l = 1 if direction == curses.KEY_LEFT else 0
        dir_r = 1 if direction == curses.KEY_RIGHT else 0
        dir_u = 1 if direction == curses.KEY_UP else 0
        dir_d = 1 if direction == curses.KEY_DOWN else 0

        state = (
            # Danger flags (wall, body, or obstacle)
            1 if self._is_collision(point_u, snake, obstacles) else 0,  # Danger Up
            1 if self._is_collision(point_d, snake, obstacles) else 0,  # Danger Down
            1 if self._is_collision(point_l, snake, obstacles) else 0,  # Danger Left
            1 if self._is_collision(point_r, snake, obstacles) else 0,  # Danger Right

            # Current direction
            dir_u, dir_d, dir_l, dir_r,

            # Food location flags
            1 if food[0] < head[0] else 0,  # Food is Up
            1 if food[0] > head[0] else 0,  # Food is Down
            1 if food[1] < head[1] else 0,  # Food is Left
            1 if food[1] > head[1] else 0,  # Food is Right
        )
        return state

    def get_action(self, state, exploration_rate):
        """Get action using epsilon-greedy strategy"""
        if random.random() < exploration_rate:
            action = random.choice(self.actions)
            logging.debug(f"Exploration action chosen: {action}")
            return action
        state_key = str(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0] * len(self.actions)
        # Find the maximum Q-value and all actions with that value
        max_q = max(self.q_table[state_key])
        best_actions = [i for i, q in enumerate(self.q_table[state_key]) if q == max_q]
        # Random tie-breaking for equally good actions
        action = self.actions[random.choice(best_actions)]
        logging.debug(f"Greedy action chosen: {action}")
        return action

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning formula"""
        state_key = str(state)
        next_state_key = str(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0] * len(self.actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0] * len(self.actions)

        action_idx = self.actions.index(action)
        current_q = self.q_table[state_key][action_idx]
        max_future_q = max(self.q_table[next_state_key])
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
        self.q_table[state_key][action_idx] = new_q
        logging.debug(f"Updated Q-table: state={state_key}, action={action}, reward={reward}, new_q={new_q}")

    def _is_collision(self, point, snake, obstacles):
        """Check if a point is a wall, part of the snake, or an obstacle."""
        # Check wall collision
        if (point[0] <= 0 or point[0] >= self.height - 1 or
            point[1] <= 0 or point[1] >= self.width - 2):
            return True
        # Check body or obstacle collision
        return point in snake or point in obstacles

    def load_q_table(self):
        """Load Q-table from file if it exists"""
        try:
            if os.path.exists('snake_qtable.pkl'):
                with open('snake_qtable.pkl', 'rb') as f:
                    self.q_table = pickle.load(f)
                logging.info("Loaded Q-table from snake_qtable.pkl")
        except Exception as e:
            self.q_table = {}
            logging.error(f"Error loading Q-table: {e}")

    def save_q_table(self):
        """Save Q-table to file"""
        try:
            with open('snake_qtable.pkl', 'wb') as f:
                pickle.dump(self.q_table, f)
            logging.info("Saved Q-table to snake_qtable.pkl")
        except Exception as e:
            logging.error(f"Error saving Q-table: {e}")
