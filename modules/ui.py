"""
UI module for menu system and user interface
"""

import curses
import time
import logging
from .config import DIFFICULTY_CONFIG, GAME_MODES
from .multiplayer_client import MultiplayerClient, test_connection


def show_startup_screen(stdscr):
    """Display startup screen and get game configuration"""
    curses.curs_set(0)
    stdscr.clear()

    # Game configuration defaults
    selected_difficulty = 2
    selected_game_mode = 1
    target_score = 100
    game_time_minutes = 5
    current_menu = 'main'  # main, difficulty, game_mode, score, time, lan

    # Validate selected difficulty
    if selected_difficulty not in DIFFICULTY_CONFIG:
        logging.error(f"Invalid difficulty selected: {selected_difficulty}")
        raise KeyError(f"Invalid difficulty key: {selected_difficulty}")

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Title
        title = "SNAKE GAME SETUP"
        stdscr.addstr(2, width // 2 - len(title) // 2, title)

        if current_menu == 'main':
            # Main menu
            stdscr.addstr(5, 4, "Current Configuration:")
            stdscr.addstr(6, 6, f"Difficulty: {DIFFICULTY_CONFIG[selected_difficulty]['name']}")
            stdscr.addstr(7, 6, f"Game Mode: {GAME_MODES[selected_game_mode]['name']}")
            if selected_game_mode == 2:
                stdscr.addstr(8, 6, f"Target Score: {target_score}")
            elif selected_game_mode == 3:
                stdscr.addstr(8, 6, f"Game Time: {game_time_minutes} minutes")

            stdscr.addstr(10, 4, "Options:")
            stdscr.addstr(11, 6, "1. Change Difficulty")
            stdscr.addstr(12, 6, "2. Change Game Mode")
            if selected_game_mode == 2:
                stdscr.addstr(13, 6, "3. Set Target Score")
            elif selected_game_mode == 3:
                stdscr.addstr(13, 6, "3. Set Game Time")
            stdscr.addstr(14, 6, "4. LAN Multiplayer")
            stdscr.addstr(16, 6, "SPACE. Start Game")
            stdscr.addstr(17, 6, "Q. Quit")

            # Controls info
            stdscr.addstr(height - 4, 4, "Controls:")
            stdscr.addstr(height - 3, 6, "Player 1: Arrow keys, SPACE for turbo")
            stdscr.addstr(height - 2, 6, "Player 2: AI or Human (Multiplayer)")

        elif current_menu == 'difficulty':
            stdscr.addstr(5, 4, "Select Difficulty:")
            for i, (key, config) in enumerate(DIFFICULTY_CONFIG.items()):
                marker = ">>>" if key == selected_difficulty else "   "
                stdscr.addstr(7 + i, 6, f"{marker} {key}. {config['name']} (Speed: {config['speed']:.2f}s)")

            stdscr.addstr(height - 3, 6, "Use number keys to select, ENTER to confirm, ESC to go back")

        elif current_menu == 'game_mode':
            stdscr.addstr(5, 4, "Select Game Mode:")
            for i, (key, mode) in enumerate(GAME_MODES.items()):
                marker = ">>>" if key == selected_game_mode else "   "
                stdscr.addstr(7 + i, 6, f"{marker} {key}. {mode['name']}")
                stdscr.addstr(8 + i, 10, f"    {mode['description']}")

            stdscr.addstr(height - 3, 6, "Use number keys to select, ENTER to confirm, ESC to go back")

        elif current_menu == 'score':
            stdscr.addstr(5, 4, "Set Target Score:")
            stdscr.addstr(7, 6, f"Current: {target_score}")
            stdscr.addstr(9, 6, "Presets:")
            presets = [50, 100, 200, 500]
            for i, preset in enumerate(presets):
                stdscr.addstr(10 + i, 8, f"{i+1}. {preset} points")

            stdscr.addstr(height - 4, 6, "Use number keys for presets")
            stdscr.addstr(height - 3, 6, "+ / - to adjust by 10, ENTER to confirm, ESC to go back")

        elif current_menu == 'time':
            stdscr.addstr(5, 4, "Set Game Time:")
            stdscr.addstr(7, 6, f"Current: {game_time_minutes} minutes")
            stdscr.addstr(9, 6, "Presets:")
            presets = [2, 5, 10, 15]
            for i, preset in enumerate(presets):
                stdscr.addstr(10 + i, 8, f"{i+1}. {preset} minutes")

            stdscr.addstr(height - 4, 6, "Use number keys for presets")
            stdscr.addstr(height - 3, 6, "+ / - to adjust by 1, ENTER to confirm, ESC to go back")

        elif current_menu == 'lan':
            # LAN Multiplayer setup
            multiplayer_client = setup_lan_multiplayer(stdscr)
            if multiplayer_client:
                logging.info("LAN multiplayer setup completed")
                return {
                    'difficulty': selected_difficulty,
                    'game_mode': selected_game_mode,
                    'target_score': target_score,
                    'game_time_minutes': game_time_minutes,
                    'multiplayer_enabled': True,
                    'multiplayer_client': multiplayer_client
                }
            else:
                current_menu = 'main'

        stdscr.refresh()

        # Handle input
        key = stdscr.getch()

        if current_menu == 'main':
            if key == ord('1'):
                current_menu = 'difficulty'
            elif key == ord('2'):
                current_menu = 'game_mode'
            elif key == ord('3'):
                if selected_game_mode == 2:
                    current_menu = 'score'
                elif selected_game_mode == 3:
                    current_menu = 'time'
            elif key == ord('4'):
                current_menu = 'lan'
            elif key == ord(' '):
                # Start game
                logging.info("Starting game with configuration: %s", {
                    'difficulty': selected_difficulty,
                    'game_mode': selected_game_mode,
                    'target_score': target_score,
                    'game_time_minutes': game_time_minutes,
                    'multiplayer_enabled': False
                })
                return {
                    'difficulty': selected_difficulty,
                    'game_mode': selected_game_mode,
                    'target_score': target_score,
                    'game_time_minutes': game_time_minutes,
                    'multiplayer_enabled': False
                }
            elif key == ord('q') or key == ord('Q'):
                logging.info("User quit from startup screen")
                return None

        elif current_menu == 'difficulty':
            if key >= ord('1') and key <= ord('5'):
                selected_difficulty = key - ord('0')
                logging.debug(f"Selected difficulty: {selected_difficulty}")
            elif key == ord('\n') or key == ord('\r'):
                current_menu = 'main'
            elif key == 27:  # ESC
                current_menu = 'main'

        elif current_menu == 'game_mode':
            if key >= ord('1') and key <= ord('3'):
                selected_game_mode = key - ord('0')
                logging.debug(f"Selected game mode: {selected_game_mode}")
            elif key == ord('\n') or key == ord('\r'):
                current_menu = 'main'
            elif key == 27:  # ESC
                current_menu = 'main'

        elif current_menu == 'score':
            if key >= ord('1') and key <= ord('4'):
                presets = [50, 100, 200, 500]
                target_score = presets[key - ord('1')]
                logging.debug(f"Set target score: {target_score}")
            elif key == ord('+') or key == ord('='):
                target_score += 10
                logging.debug(f"Increased target score to: {target_score}")
            elif key == ord('-'):
                target_score = max(10, target_score - 10)
                logging.debug(f"Decreased target score to: {target_score}")
            elif key == ord('\n') or key == ord('\r'):
                current_menu = 'main'
            elif key == 27:  # ESC
                current_menu = 'main'

        elif current_menu == 'time':
            if key >= ord('1') and key <= ord('4'):
                presets = [2, 5, 10, 15]
                game_time_minutes = presets[key - ord('1')]
                logging.debug(f"Set game time: {game_time_minutes} minutes")
            elif key == ord('+') or key == ord('='):
                game_time_minutes += 1
                logging.debug(f"Increased game time to: {game_time_minutes}")
            elif key == ord('-'):
                game_time_minutes = max(1, game_time_minutes - 1)
                logging.debug(f"Decreased game time to: {game_time_minutes}")
            elif key == ord('\n') or key == ord('\r'):
                current_menu = 'main'
            elif key == 27:  # ESC
                current_menu = 'main'


def setup_lan_multiplayer(stdscr):
    """Setup LAN multiplayer configuration"""
    curses.curs_set(1)
    _, width = stdscr.getmaxyx()

    while True:
        stdscr.clear()
        stdscr.addstr(2, 4, "LAN Multiplayer Setup")
        stdscr.addstr(4, 4, "Choose your role:")
        stdscr.addstr(6, 6, "1. Host a Game")
        stdscr.addstr(7, 8, "   - Start server and wait for opponent")
        stdscr.addstr(8, 8, "   - Your IP will be displayed")
        stdscr.addstr(10, 6, "2. Join a Game")
        stdscr.addstr(11, 8, "   - Connect to another player's server")
        stdscr.addstr(12, 8, "   - Enter their IP address")
        stdscr.addstr(14, 6, "ESC. Back to Main Menu")
        stdscr.refresh()

        key = stdscr.getch()

        if key == ord('1'):
            # Host setup
            stdscr.clear()
            stdscr.addstr(2, 4, "Setting up as Host...")
            stdscr.addstr(4, 4, "Choose opponent type:")
            stdscr.addstr(6, 6, "1. Human opponent")
            stdscr.addstr(7, 6, "2. AI opponent")
            stdscr.addstr(9, 6, "ESC to go back")
            stdscr.refresh()

            opponent_key = stdscr.getch()
            if opponent_key == ord('1'):
                opponent_type = "human"
            elif opponent_key == ord('2'):
                opponent_type = "ai"
            elif opponent_key == 27:  # ESC
                continue
            else:
                continue

            from .utils import start_server_background
            if not start_server_background():
                stdscr.addstr(10, 4, "Failed to start multiplayer server!")
                stdscr.addstr(11, 4, "Press any key to continue...")
                stdscr.getch()
                logging.error("Failed to start multiplayer server")
                continue
            time.sleep(2)  # Give server time to start
            try:
                client = MultiplayerClient("localhost")
                if not client.connect_websocket():
                    stdscr.addstr(10, 4, "Failed to connect to server!")
                    stdscr.addstr(11, 4, "Press any key to continue...")
                    stdscr.getch()
                    logging.error("Failed to connect WebSocket for host")
                    continue

                session_data = client.create_session(p1_type="human", p2_type=opponent_type)
                if not session_data:
                    stdscr.addstr(10, 4, "Failed to create session!")
                    stdscr.addstr(11, 4, "Press any key to continue...")
                    stdscr.getch()
                    logging.error("Failed to create multiplayer session")
                    continue

                stdscr.clear()
                stdscr.addstr(2, 4, "Hosting Game")
                stdscr.addstr(4, 4, f"Your IP: {client.get_local_ip()}")
                stdscr.addstr(5, 4, f"Session ID: {session_data['session_id']}")
                stdscr.addstr(7, 4, "Waiting for opponent to join...")
                stdscr.addstr(9, 4, "Press ESC to cancel")
                stdscr.refresh()

                start_time = time.time()
                while time.time() - start_time < 30:  # Wait up to 30 seconds
                    message = client.get_message()
                    if message and message.get('type') == 'player_joined':
                        stdscr.addstr(8, 4, "Opponent joined! Starting game...")
                        stdscr.refresh()
                        time.sleep(2)
                        logging.info(f"Host created session {session_data['session_id']} with opponent type {opponent_type}")
                        return client
                    elif stdscr.getch() == 27:  # ESC
                        client.disconnect()
                        stdscr.addstr(8, 4, "Cancelled hosting")
                        stdscr.addstr(9, 4, "Press any key to continue...")
                        stdscr.getch()
                        logging.info("Host cancelled session")
                        return None
                    time.sleep(0.1)

                client.disconnect()
                stdscr.addstr(8, 4, "No opponent joined in time")
                stdscr.addstr(9, 4, "Press any key to continue...")
                stdscr.getch()
                logging.warning("No opponent joined within timeout")
                return None

            except Exception as e:
                stdscr.addstr(10, 4, f"Error: {str(e)}")
                stdscr.addstr(11, 4, "Press any key to continue...")
                stdscr.getch()
                logging.error(f"Error setting up host: {e}")
                return None

        elif key == ord('2'):
            # Join setup
            stdscr.clear()
            stdscr.addstr(2, 4, "Join a Game")
            stdscr.addstr(4, 4, "Enter server IP address:")
            stdscr.addstr(6, 6, "> ")
            stdscr.refresh()

            server_ip = ""
            curses.echo()
            while True:
                char = stdscr.getch()
                if char == ord('\n') or char == ord('\r'):
                    break
                elif char == 27:  # ESC
                    curses.noecho()
                    return None
                # Printable characters, check if we have space to add more
                elif char >= 32 and char <= 126 and len(server_ip) < width - 10:
                    server_ip += chr(char)
                    stdscr.addstr(6, 8, server_ip)
                    stdscr.refresh()
                elif char == curses.KEY_BACKSPACE or char == 127:
                    server_ip = server_ip[:-1] if server_ip else ""
                    stdscr.addstr(6, 8, server_ip + " ")
                    stdscr.refresh()
            curses.noecho()

            if not server_ip:
                stdscr.addstr(8, 4, "No IP entered")
                stdscr.addstr(9, 4, "Press any key to continue...")
                stdscr.getch()
                logging.warning("No server IP entered")
                continue

            stdscr.addstr(8, 4, f"Connecting to {server_ip}...")
            stdscr.refresh()

            if not test_connection(server_ip):
                stdscr.addstr(9, 4, "Failed to connect to server!")
                stdscr.addstr(10, 4, "Press any key to continue...")
                stdscr.getch()
                logging.error(f"Failed to connect to server {server_ip}")
                continue

            try:
                client = MultiplayerClient(server_ip)
                server_info = client.get_server_info()
                if not server_info:
                    stdscr.addstr(9, 4, "Failed to get server info!")
                    stdscr.addstr(10, 4, "Press any key to continue...")
                    stdscr.getch()
                    logging.error("Failed to get server info")
                    continue

                sessions = client.get(f"{client.base_url}/sessions").json()['sessions']
                if not sessions:
                    stdscr.addstr(9, 4, "No active sessions found!")
                    stdscr.addstr(10, 4, "Press any key to continue...")
                    stdscr.getch()
                    logging.warning("No active sessions found")
                    continue

                stdscr.clear()
                stdscr.addstr(2, 4, "Available Sessions:")
                for i, session in enumerate(sessions):
                    if session['active'] and session['players_count'] < 2:
                        stdscr.addstr(4 + i, 6, f"{i+1}. Session {session['session_id']} (Host: {session['host_id'][:8]}...)")
                stdscr.addstr(4 + len(sessions), 6, "Enter session number or ESC to go back")
                stdscr.refresh()

                session_num = stdscr.getch()
                if session_num == 27:  # ESC
                    continue
                session_num = session_num - ord('1')
                if session_num < 0 or session_num >= len(sessions):
                    stdscr.addstr(6 + len(sessions), 4, "Invalid session number")
                    stdscr.addstr(7 + len(sessions), 4, "Press any key to continue...")
                    stdscr.getch()
                    logging.warning("Invalid session number entered")
                    continue

                session_id = sessions[session_num]['session_id']
                if not client.connect_websocket():
                    stdscr.addstr(6 + len(sessions), 4, "Failed to connect WebSocket!")
                    stdscr.addstr(7 + len(sessions), 4, "Press any key to continue...")
                    stdscr.getch()
                    logging.error("Failed to connect WebSocket for join")
                    continue

                if client.join_session(session_id):
                    stdscr.addstr(6 + len(sessions), 4, "Joined session successfully!")
                    stdscr.addstr(7 + len(sessions), 4, "Starting game...")
                    stdscr.refresh()
                    time.sleep(2)
                    logging.info(f"Joined session {session_id} on server {server_ip}")
                    return client
                else:
                    stdscr.addstr(6 + len(sessions), 4, "Failed to join session!")
                    stdscr.addstr(7 + len(sessions), 4, "Press any key to continue...")
                    stdscr.getch()
                    logging.error(f"Failed to join session {session_id}")
                    continue

            except Exception as e:
                stdscr.addstr(9, 4, f"Error: {str(e)}")
                stdscr.addstr(10, 4, "Press any key to continue...")
                stdscr.getch()
                logging.error(f"Error joining game: {e}")
                continue

        elif key == 27:  # ESC
            curses.curs_set(0)
            return None


def show_countdown(stdscr):
    """Show countdown before game starts"""
    stdscr.clear()
    height, width = stdscr.getmaxyx()

    stdscr.addstr(height // 2 - 2, width // 2 - 10, "Game starting in...")

    for i in range(3, 0, -1):
        stdscr.addstr(height // 2, width // 2 - 1, str(i))
        stdscr.refresh()
        time.sleep(1)

    stdscr.addstr(height // 2, width // 2 - 10, "GO!")
    stdscr.refresh()
    time.sleep(0.5)
    logging.info("Game countdown completed")
