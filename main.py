#!/usr/bin/env python3
"""
Main entry point for the AI-powered Snake Game

This file initializes the game, handles the startup sequence,
and coordinates between UI, multiplayer, and game modules.
"""

import curses
import sys
import time
import logging
import os

# Add the project root to the Python path to resolve module imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.ui import show_startup_screen, show_countdown
from modules.game import GameState, GameEngine
from modules import config

# Initialize logging
logging.basicConfig(
    filename='snake_game.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)


def print_final_results(game_state, config_data):
    """Print final game results to terminal after curses cleanup"""
    try:
        print("\n" + "="*50)
        print("SNAKE GAME RESULTS")
        print("="*50)

        print(f"\nGame Mode: {config.GAME_MODES[config_data['game_mode']]['name']}")
        print(f"Difficulty: {config.DIFFICULTY_CONFIG[config_data['difficulty']]['name']}")  # Ensure 'difficulty' is valid

        if config_data.get('multiplayer_enabled', False):
            print(f"\nMultiplayer Session: {config_data.get('session_id', 'N/A')}")

        print(f"\nFinal Scores:")
        print(f"  Player 1: {game_state.p1_score}")
        print(f"  Player 2: {game_state.p2_score}")

        if game_state.p1_score > game_state.p2_score:
            print(f"\nPlayer 1 Wins!")
        elif game_state.p2_score > game_state.p1_score:
            print(f"\nPlayer 2 Wins!")
        else:
            print(f"\nIt's a Tie!")

        print(f"\nGame Duration: {time.strftime('%M:%S', time.gmtime(game_state.game_duration))}")

        # Show win condition details
        if game_state.game_over:
            if game_state.win_condition == 'target_score':
                print(f"\nWin Condition: Target score of {config_data['target_score']} reached")
            elif game_state.win_condition == 'time_limit':
                print(f"\nWin Condition: Time limit of {config_data['game_time_minutes']} minutes reached")
            elif game_state.win_condition == 'collision':
                print(f"\nGame ended due to collision")

        print("\nThanks for playing!")
        print("="*50)
    except KeyError as e:
        logging.error(f"Configuration key error: {str(e)}")
        print(f"\nAn error occurred: Missing configuration key {str(e)}")
        sys.exit(1)


def main():
    """Main game entry point"""

    def game_wrapper(stdscr):
        # Get configuration from UI
        config_data = show_startup_screen(stdscr)
        if config_data is None:
            return  # User quit

        # Show countdown
        show_countdown(stdscr)

        # Initialize game with configuration
        height, width = stdscr.getmaxyx()
        game_state = GameState(config_data, height, width)
        game_engine = GameEngine(game_state)

        # Run the game
        game_engine.run(stdscr)

        # Print results after game ends
        print_final_results(game_state, config_data)

    # Start curses application
    try:
        curses.wrapper(game_wrapper)
    except curses.error as e:
        logging.error(f"Curses error: {e}")
        print("Error initializing screen. Your terminal may not support curses.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
