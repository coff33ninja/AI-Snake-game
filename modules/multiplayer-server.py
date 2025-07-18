#!/usr/bin/env python3
"""
Snake Game Multiplayer Server Launcher
=====================================

This script provides a clean entry point for running the Snake Game multiplayer server.
It launches the FastAPI server using uvicorn with proper configuration and displays
helpful startup information for users.

Usage:
    python modules/multiplayer_server.py

The server will start on port 8000 and display the local IP address that other
players can use to connect to multiplayer games.
"""

import sys
import uvicorn
import argparse
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import json
import time
import random
import logging

from .utils import get_local_ip
from .ai import SnakeGameAI
from .dqn_agent import DQNAgent
from .config import GAME_MODES, DIFFICULTY_CONFIG

# --- Server Configuration ---
SERVER_CONFIG = {"with_ai": False}

app = FastAPI(title="Snake Game Multiplayer Server", version="1.0.0")
ai_router = APIRouter(prefix="/ai", tags=["AI"])


class PlayerMove(BaseModel):
    player_id: str
    direction: int
    timestamp: float


class GameState(BaseModel):
    p1_snake: List[List[int]]
    p2_snake: List[List[int]]
    foods: List[List[int]]
    obstacles: List[List[int]]
    p1_score: int
    p2_score: int
    p1_game_over: bool
    p2_game_over: bool
    game_won: bool
    winner: Optional[str]
    timestamp: float
    game_start_time: float
    config: Dict[str, Any]


class AIStateInfo(BaseModel):
    """State information sent from the client to get an AI move."""
    state: List[int]  # The 11-element state vector


class AITrainingData(BaseModel):
    """Training data for the simple Q-Learning AI."""
    state: tuple
    action: int
    reward: int
    next_state: tuple
    done: bool


class DQNTrainingData(BaseModel):
    """Training data for the DQN agent."""
    state: List[float]
    action: int
    reward: float
    next_state: List[float]


class GameSession(BaseModel):
    session_id: str
    host_id: str
    client_id: Optional[str]
    p1_type: str  # "human" or "ai"
    p2_type: str  # "human" or "ai"
    ai_model: Optional[str] = None  # "q_learning" or "dqn"
    game_state: Optional[GameState]
    active: bool
    created_at: float


class ConnectionManager:

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, GameSession] = {}
        self.ai_agents: Dict[str, Any] = {}  # Can store SnakeGameAI or DQNAgent

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logging.info(f"Client {client_id} connected.")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logging.info(f"Client {client_id} disconnected.")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except:
                self.disconnect(client_id)

    async def send_to_session(self, message: str, session_id: str):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            tasks = []
            if session.host_id in self.active_connections:
                tasks.append(self.send_personal_message(message, session.host_id))
            if session.client_id and session.client_id in self.active_connections:
                tasks.append(self.send_personal_message(message, session.client_id))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def find_session_by_client_id(self, client_id: str) -> Optional[Tuple[str, GameSession]]:
        """Find the session and session_id a client belongs to."""
        for sid, session in self.sessions.items():
            if session.host_id == client_id or session.client_id == client_id:
                return sid, session
        return None, None

    def cleanup_session(self, session_id: str):
        """Removes a session and any associated AI agent, saving its progress."""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            logging.info(f"Removed session {session_id}.")

            if session_id in self.ai_agents:
                agent = self.ai_agents.pop(session_id)
                logging.info(f"Saving model for AI in session {session_id}...")
                if isinstance(agent, SnakeGameAI):
                    agent.save_q_table()
                    logging.info(f"Saved Q-table for session {session_id}.")
                elif isinstance(agent, DQNAgent):
                    # Save model with a session-specific name to avoid conflicts
                    model_path = f"dqn_model_{session.host_id}.pth"
                    agent.save_model(path=model_path)
                    logging.info(f"Saved DQN model to {model_path}.")

    async def handle_disconnect(self, client_id: str):
        """Handle all logic related to a client disconnecting."""
        self.disconnect(client_id)  # Remove from active connections first

        session_id, session = self.find_session_by_client_id(client_id)

        if not session:
            logging.info(f"Disconnected client {client_id} was not in any session.")
            return

        # Case 1: The host disconnected. The session must be terminated.
        if session.host_id == client_id:
            logging.info(f"Host {client_id} disconnected from session {session_id}. Ending session.")
            message = json.dumps({"type": "session_ended", "reason": "Host disconnected."})

            # Notify the other player (if they exist and are still connected)
            if session.client_id and session.client_id in self.active_connections:
                await self.send_personal_message(message, session.client_id)

            # Remove the session entirely
            self.cleanup_session(session_id)

        # Case 2: The guest/client disconnected. The session remains, but is now open.
        elif session.client_id == client_id:
            logging.info(f"Client {client_id} disconnected from session {session_id}. Notifying host.")
            session.client_id = None  # Mark the spot as vacant
            message = json.dumps({"type": "player_disconnected", "client_id": client_id})
            # Notify the host (if they are still connected)
            if session.host_id in self.active_connections:
                await self.send_personal_message(message, session.host_id)


manager = ConnectionManager()


@app.get("/")
async def read_root():
    local_ip = get_local_ip()
    return {
        "message": "Snake Game Multiplayer Server",
        "version": "1.0.0",
        "server_ip": local_ip,
        "websocket_url": f"ws://{local_ip}:8000/ws/",
        "active_sessions": len(manager.sessions),
        "active_connections": len(manager.active_connections)
    }


@app.get("/server_info")
async def get_server_info():
    """Get server information including IP and port"""
    local_ip = get_local_ip()
    return {
        "ip": local_ip,
        "port": 8000,
        "websocket_url": f"ws://{local_ip}:8000/ws/",
        "active_sessions": len(manager.sessions),
        "active_connections": len(manager.active_connections)
    }


@app.post("/create_session")
async def create_session(host_id: str, p1_type: str="human", p2_type: str="human", ai_model: Optional[str]=None):
    """Create a new game session"""
    session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"

    # Basic session object
    session = GameSession(
        session_id=session_id,
        host_id=host_id,
        client_id=None,
        p1_type=p1_type,
        p2_type=p2_type,
        game_state=None,
        ai_model=ai_model,
        active=True,
        created_at=time.time()
    )

    manager.sessions[session_id] = session

    # If the session is against an AI, create and store the AI agent instance
    if p2_type == "ai":
        if not SERVER_CONFIG["with_ai"]:
            del manager.sessions[session_id]  # Clean up before raising
            logging.error("Attempted to create AI session while AI hosting is disabled.")
            raise HTTPException(status_code=403, detail="Server not configured to host AI. Start with --with-ai flag.")

        if ai_model == "q_learning":
            agent = SnakeGameAI()
            agent.load_q_table()  # Load existing knowledge
            manager.ai_agents[session_id] = agent
            logging.info(f"Created Q-Learning AI agent for session {session_id}")
        elif ai_model == "dqn":
            agent = DQNAgent(state_dim=11, action_dim=4)  # Assuming 11 state features, 4 actions
            agent.load_model(path=f"dqn_model_{host_id}.pth")  # Load player-specific model
            manager.ai_agents[session_id] = agent
            logging.info(f"Created DQN AI agent for session {session_id}")

    logging.info(f"Created session {session_id} for host {host_id}")
    return {"status": "created", "session": session.dict()}


@app.post("/join_session")
async def join_session(session_id: str, client_id: str):
    """Join an existing game session"""
    if session_id not in manager.sessions:
        logging.error(f"Session {session_id} not found")
        raise HTTPException(status_code=404, detail="Session not found")

    session = manager.sessions[session_id]
    if session.client_id is not None:
        logging.error(f"Session {session_id} is full")
        raise HTTPException(status_code=400, detail="Session is full")
    if session.p2_type == "ai":
        logging.error(f"Attempt to join an AI session {session_id}")
        raise HTTPException(status_code=403, detail="Cannot join a session hosted by an AI opponent.")

    session.client_id = client_id

    # Notify the host that a player has joined
    message = {"type": "player_joined", "data": {"client_id": client_id, "session_id": session_id}}
    await manager.send_personal_message(json.dumps(message), session.host_id)

    # Also notify the new client that they have joined successfully
    await manager.send_personal_message(json.dumps({"type": "join_success", "data": session.dict()}), client_id)
    logging.info(f"Client {client_id} joined session {session_id}")
    return {"status": "joined", "session": session.dict()}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id not in manager.sessions:
        logging.error(f"Session {session_id} not found")
        raise HTTPException(status_code=404, detail="Session not found")

    return manager.sessions[session_id]


@app.post("/update_game_state")
async def update_game_state(session_id: str, game_state: GameState):
    """Update the game state for a session"""
    if session_id not in manager.sessions:
        logging.error(f"Session {session_id} not found")
        raise HTTPException(status_code=404, detail="Session not found")

    manager.sessions[session_id].game_state = game_state

    # Broadcast updated game state to all players in the session
    message = {
        "type": "game_state_update",
        "data": game_state.dict()
    }
    await manager.send_to_session(json.dumps(message), session_id)
    logging.debug(f"Updated game state for session {session_id}")

    return {"status": "updated"}


@app.post("/player_move")
async def player_move(session_id: str, move: PlayerMove):
    """Handle player movement"""
    if session_id not in manager.sessions:
        logging.error(f"Session {session_id} not found")
        raise HTTPException(status_code=404, detail="Session not found")

    # Broadcast player move to all players in the session
    message = {
        "type": "player_move",
        "data": move.dict()
    }
    await manager.send_to_session(json.dumps(message), session_id)
    logging.debug(f"Sent player move for session {session_id}: {move.dict()}")

    return {"status": "move_sent"}


@ai_router.post("/q/move/{session_id}")
async def get_q_learning_move(session_id: str, state_info: AIStateInfo):
    if session_id not in manager.ai_agents or not isinstance(manager.ai_agents[session_id], SnakeGameAI):
        raise HTTPException(status_code=404, detail="Q-Learning AI agent not found for this session.")
    agent = manager.ai_agents[session_id]
    action = agent.get_action(state_info.state)
    return {"action": action}


@ai_router.post("/q/train/{session_id}")
async def train_q_learning_step(session_id: str, data: AITrainingData):
    if session_id not in manager.ai_agents or not isinstance(manager.ai_agents[session_id], SnakeGameAI):
        raise HTTPException(status_code=404, detail="Q-Learning AI agent not found for this session.")
    agent = manager.ai_agents[session_id]
    agent.train_step(data.state, data.action, data.reward, data.next_state, data.done)
    return {"status": "training_step_received"}


@ai_router.post("/dqn/move/{session_id}")
async def get_dqn_move(session_id: str, state_info: AIStateInfo):
    if session_id not in manager.ai_agents or not isinstance(manager.ai_agents[session_id], DQNAgent):
        raise HTTPException(status_code=404, detail="DQN AI agent not found for this session.")
    agent: DQNAgent = manager.ai_agents[session_id]

    # The agent's select_action returns a tensor, e.g., tensor([[1]])
    action_tensor = agent.select_action(state_info.state)
    action = action_tensor.item()  # Convert to a standard Python integer

    return {"action": action, "action_tensor": action_tensor.tolist()}


@ai_router.post("/dqn/train/{session_id}")
async def train_dqn_step(session_id: str, data: DQNTrainingData):
    if session_id not in manager.ai_agents or not isinstance(manager.ai_agents[session_id], DQNAgent):
        raise HTTPException(status_code=404, detail="DQN AI agent not found for this session.")
    agent: DQNAgent = manager.ai_agents[session_id]

    # Convert lists back to tensors for the agent
    state_tensor = torch.tensor(data.state, dtype=torch.float32, device=agent.device)
    action_tensor = torch.tensor([[data.action]], device=agent.device, dtype=torch.long)
    reward_tensor = torch.tensor([data.reward], device=agent.device)
    next_state_tensor = torch.tensor(data.next_state, dtype=torch.float32, device=agent.device)

    # Push to memory
    agent.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)

    # Perform one step of optimization
    agent.optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = agent.target_net.state_dict()
    policy_net_state_dict = agent.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * 0.005 + target_net_state_dict[key] * (1 - 0.005)
    agent.target_net.load_state_dict(target_net_state_dict)

    return {"status": "training_step_received"}


@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a game session"""
    if session_id not in manager.sessions:
        logging.error(f"Session {session_id} not found")
        raise HTTPException(status_code=404, detail="Session not found")

    # Notify all players that session is ending
    message = {
        "type": "session_ended",
        "data": {"session_id": session_id}
    }
    await manager.send_to_session(json.dumps(message), session_id)

    manager.cleanup_session(session_id)
    logging.info(f"Ended session {session_id}")

    return {"status": "session_ended"}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": [
            {
                "session_id": sid,
                "host_id": session.host_id,
                "client_id": session.client_id,
                "p1_type": session.p1_type,
                "p2_type": session.p2_type,
                "active": session.active,
                "created_at": session.created_at,
                "players_count": 1 + (1 if session.client_id else 0)
            }
            for sid, session in manager.sessions.items()
        ]
    }


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket, client_id)

    try:
        await manager.send_personal_message(
            json.dumps({"type": "connected", "client_id": client_id}),
            client_id
        )

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Forward game-related messages to the other player in the session
            session_id = message.get("session_id")
            if session_id and session_id in manager.sessions:
                session = manager.sessions[session_id]

                # Determine the recipient
                recipient_id = None
                if client_id == session.host_id:
                    recipient_id = session.client_id
                elif client_id == session.client_id:
                    recipient_id = session.host_id

                # Forward the message if the recipient is connected
                if recipient_id and recipient_id in manager.active_connections:
                    await manager.send_personal_message(data, recipient_id)

    except WebSocketDisconnect:
        await manager.handle_disconnect(client_id)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


# Include the AI router in the main application
app.include_router(ai_router)


def main():
    """Main entry point for the multiplayer server launcher."""

    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Snake Game Multiplayer Server")
    parser.add_argument('--with-ai', action='store_true', help="Enable the server to host AI opponents.")
    args = parser.parse_args()

    SERVER_CONFIG["with_ai"] = args.with_ai
    local_ip = get_local_ip()

    print("=" * 50)
    print("SNAKE GAME MULTIPLAYER SERVER")
    print("=" * 50)
    print()
    print("🚀 Starting multiplayer server...")
    print(f"📡 Server IP: {local_ip}")
    print(f"🌐 Port: 8000")
    if SERVER_CONFIG["with_ai"]:
        print("🤖 AI Hosting: ENABLED")
    else:
        print("🤖 AI Hosting: DISABLED (run with --with-ai to enable)")
    print()
    print("🔗 Connection Details:")
    print(f"   WebSocket URL: ws://{local_ip}:8000/ws/")
    print(f"   API Base URL: http://{local_ip}:8000")
    print()
    print("🎮 Share this information with other players!")
    print("   Players can connect using the IP address above")
    print()
    print("📋 Available Endpoints:")
    print("   GET  /health           - Health check")
    print("   GET  /sessions         - List active sessions")
    print("   POST /create_session   - Create new game session")
    print("   POST /join_session     - Join existing session")
    print("   WebSocket /ws/{client_id} - Real-time communication")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    try:
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        print("👋 Thanks for playing Snake Game!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("💡 Make sure port 8000 is not already in use")
        sys.exit(1)


if __name__ == "__main__":
    main()
