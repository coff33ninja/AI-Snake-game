 # AI Snake Game

 A modern, terminal-based Snake game with a twist. Play against a Q-learning AI opponent in a split-screen setup or challenge a friend over your local network. This project demonstrates core game development principles, AI/ML integration, and client-server networking in Python.

 ## ✨ Features

 - **Intelligent AI Opponent**: Battle against a snake controlled by a Q-learning algorithm that learns and improves over time.
 - **Split-Screen View**: Watch the human player and the AI compete side-by-side in real-time, right in your terminal.
 - **LAN Multiplayer**: Host a game and have a friend join over the local network for a head-to-head challenge.
 - **Multiple Game Modes**:
   - **First to Score**: Race to be the first to reach a target score.
   - **Time Attack**: Compete for the highest score before the time runs out.
 - **Adjustable Difficulty**: Choose from Easy, Normal, or Hard settings to change the game speed.
 - **Turbo Boost**: Use a temporary speed boost to outmaneuver your opponent, with a strategic cooldown.
 - **Persistent AI Learning**: The AI's "brain" (Q-table) is saved to `snake_qtable.pkl` and loaded across sessions, allowing it to learn from every game.

 ## 📋 Requirements

 - Python 3.7+
 - `requests`
 - `websocket-client`
 - `fastapi`
 - `uvicorn`
 - `curses` (standard on Linux/macOS; for Windows, use `windows-curses`)

 ## 🚀 Installation

 1.  **Clone the repository:**
     ```bash
     git clone (https://github.com/coff33ninja/AI-Snake-game)
     cd AI-Snake-game
     ```

 2.  **Install the dependencies:**
     It's highly recommended to use a virtual environment.
     ```bash
     pip install -r requirements.txt
     ```
     *Note: The `requirements.txt` is configured to automatically install `windows-curses` for Windows users. This dependency is not needed and will be ignored on macOS and Linux.*

 ## 🎮 How to Play

 ### Single Player (vs. AI)

 Run the main game file. The menu will allow you to configure the game settings.

 ```bash
 python main.py
 ```

 Use the on-screen menu to select your difficulty and game mode, then start the game.

 ### Multiplayer (LAN)

 Multiplayer requires one person to run the server and two people (or one person opening two clients) to act as players.

 1.  **Start the Server:**
     On one machine, run the multiplayer server script. It will display its local IP address, which other players will need.
     ```bash
     python modules/multiplayer-server.py
     ```

 2.  **Host Player:**
     - Run `python main.py`.
     - In the menu, navigate to `4. LAN Multiplayer` -> `1. Host a Game`.
     - The game will display the session info and wait for another player to join.

 3.  **Joining Player:**
     - Run `python main.py`.
     - Navigate to `4. LAN Multiplayer` -> `2. Join a Game`.
     - Enter the IP address of the server machine (from step 1).
     - Select the available session to join the game.

 ## ⌨️ Controls

 - **Arrow Keys**: Control the snake's direction.
 - **Spacebar**: Activate Turbo Boost (when available).
 - **Q**: Quit the game at any time.

 ## 📂 Project Structure

 ```
 AI-Snake-game/
 ├── main.py                 # Main entry point for the game client
 ├── modules/
 │   ├── __init__.py
 │   ├── ai.py                 # Q-learning AI logic and model management
 │   ├── config.py             # Game constants and configuration
 │   ├── game.py               # Core game logic, state, and engine
 │   ├── multiplayer-server.py # FastAPI/WebSocket server for multiplayer
 │   ├── multiplayer_client.py # Client-side logic for LAN play
 │   ├── ui.py                 # Curses-based user interface and menus
 │   └── utils.py              # Shared utility functions (e.g., get IP)
 ├── snake_qtable.pkl        # Saved state of the AI's learned Q-table
 └── snake_game.log          # Log file for debugging
 ```

 ## 🧠 AI Implementation

 The AI opponent uses a **Q-learning** model, a type of reinforcement learning.

 - **State**: The state is a simplified, 12-element tuple representing the AI's immediate surroundings. It includes binary flags for danger (up, down, left, right), current direction, and the relative location of the food.
 - **Actions**: The AI can choose one of four actions: move up, down, left, or right.
 - **Reward System**: The AI receives positive rewards for eating food and negative rewards (penalties) for colliding with walls, obstacles, or itself.
 - **Learning**: The Q-table, which maps state-action pairs to expected rewards, is updated after each move using the Bellman equation. This allows the AI to learn optimal strategies over many games.

 ## 📜 License

 This project is licensed under the MIT License. See the LICENSE file for details.
