"""
Game configuration and constants
"""

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01

# Turbo feature parameters
TURBO_SPEED_MULTIPLIER = 3.0
TURBO_DURATION = 3.0  # seconds
TURBO_COOLDOWN = 15.0  # seconds

# Level system parameters
LEVEL_UP_SCORE = 50  # Points needed to advance to next level
MAX_LEVEL = 5

# Game mode configurations
DIFFICULTY_CONFIG = {
    1: {'speed': 0.20, 'obstacles': [], 'food_count': 1, 'name': 'Easy', 'ai_exploration': 0.3},
    2: {'speed': 0.15, 'obstacles': [], 'food_count': 1, 'name': 'Medium', 'ai_exploration': 0.2},
    3: {'speed': 0.12, 'obstacles': [[10, 10], [10, 11], [10, 12]], 'food_count': 1, 'name': 'Hard', 'ai_exploration': 0.1},
    4: {'speed': 0.10, 'obstacles': [[10, 10], [10, 11], [10, 12], [15, 20], [15, 21]], 'food_count': 2, 'name': 'Expert', 'ai_exploration': 0.05},
    5: {'speed': 0.08, 'obstacles': [[10, 10], [10, 11], [10, 12], [15, 20], [15, 21], [25, 15], [25, 16]], 'food_count': 2, 'name': 'Insane', 'ai_exploration': 0.01}
}

# Game modes
GAME_MODES = {
    1: {'name': 'Classic', 'description': 'Play until both players die'},
    2: {'name': 'First to Score', 'description': 'First player to reach target score wins'},
    3: {'name': 'Time Attack', 'description': 'Highest score when time runs out wins'}
}

# Reward values
REWARD_FOOD = 20
REWARD_FOOD_P1 = -2
REWARD_COLLISION = -10

# Curses settings
CURSES_TIMEOUT = 150
