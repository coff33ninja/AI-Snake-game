"""
Multiplayer client module for LAN gameplay
"""
import logging
import requests
import json
import uuid
import threading
import websocket  # pip install websocket-client
from queue import Queue, Empty


class MultiplayerClient:

    def __init__(self, server_ip, server_port=8000):
        self.server_ip = server_ip
        self.server_port = server_port
        self.base_url = f"http://{server_ip}:{server_port}"
        self.client_id = str(uuid.uuid4())
        self.session_id = None
        self.is_host = False
        self.ws = None
        self.ws_thread = None
        self.ws_connected = False
        # Thread-safe queue to pass messages from the ws_thread to the main game loop
        self.message_queue = Queue()

    def get_server_info(self):
        """Get server information"""
        try:
            response = requests.get(f"{self.base_url}/server_info", timeout=5)
            if response.status_code == 200:
                return response.json()
            logging.error(f"Failed to get server info. Status: {response.status_code}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting server info: {e}")
            return None

    def create_session(self, p1_type="human", p2_type="human"):
        """Create a new game session"""
        try:
            response = requests.post(
                f"{self.base_url}/create_session",
                params={
                    "host_id": self.client_id,
                    "p1_type": p1_type,
                    "p2_type": p2_type
                },
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                self.session_id = data["session_id"]
                self.is_host = True
                logging.info(f"Created session: {self.session_id}")
                return data
            logging.error(f"Failed to create session. Status: {response.status_code}, Body: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error creating session: {e}")
            return None

    def join_session(self, session_id):
        """Join an existing game session"""
        try:
            response = requests.post(
                f"{self.base_url}/join_session",
                params={
                    "session_id": session_id,
                    "client_id": self.client_id
                },
                timeout=5
            )
            if response.status_code == 200:
                self.session_id = session_id
                self.is_host = False
                logging.info(f"Joined session: {session_id}")
                return response.json()
            logging.error(f"Failed to join session. Status: {response.status_code}, Body: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error joining session: {e}")
            return None

    def send_message(self, message_data):
        """Sends a JSON message over the WebSocket."""
        if not self.ws_connected or not self.ws:
            logging.warning("Cannot send message, WebSocket is not connected.")
            return False
        try:
            # Ensure session_id is part of the message payload
            message_data['session_id'] = self.session_id
            self.ws.send(json.dumps(message_data))
            logging.debug(f"Sent WebSocket message: {message_data}")
            return True
        except Exception as e:
            logging.error(f"Error sending WebSocket message: {e}")
            self.ws_connected = False
            return False

    def _ws_reader(self):
        """Listens for messages on the WebSocket connection."""
        while self.ws_connected:
            try:
                message = self.ws.recv()
                if message:
                    logging.info(f"WebSocket received: {message}")
                    data = json.loads(message)
                    self.message_queue.put(data)
            except websocket.WebSocketConnectionClosedException:
                logging.info("WebSocket connection closed.")
                self.ws_connected = False
                break
            except Exception as e:
                logging.error(f"Error in WebSocket reader thread: {e}")
                self.ws_connected = False
                break

    def get_message(self):
        """Retrieves a message from the queue if available (non-blocking)."""
        try:
            return self.message_queue.get_nowait()
        except Empty:
            return None

    def connect_websocket(self):
        """Establishes a WebSocket connection to the server."""
        ws_url = f"ws://{self.server_ip}:{self.server_port}/ws/{self.client_id}"
        try:
            self.ws = websocket.create_connection(ws_url, timeout=5)
            self.ws_connected = True
            self.ws_thread = threading.Thread(target=self._ws_reader, daemon=True)
            self.ws_thread.start()
            logging.info(f"WebSocket connection established to {ws_url}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect WebSocket: {e}")
            return False

    def disconnect(self):
        """Disconnect from the server and close connections."""
        if self.ws_connected:
            self.ws_connected = False
            self.ws.close()
            if self.ws_thread:
                self.ws_thread.join(timeout=1)

        if self.session_id:
            try:
                requests.delete(f"{self.base_url}/session/{self.session_id}", timeout=5)
                logging.info(f"Session {self.session_id} ended.")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error ending session on disconnect: {e}")
        self.session_id = None
        self.is_host = False


def test_connection(server_ip, server_port=8000):
    """Test connection to the multiplayer server"""
    try:
        response = requests.get(f"http://{server_ip}:{server_port}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        logging.error(f"Failed to connect to server {server_ip}:{server_port}")
        return False
