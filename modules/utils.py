"""
Shared utility functions for the Snake Game.
"""
import socket
import logging


def get_local_ip():
    """Get the local IP address of the machine."""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        logging.debug(f"Retrieved local IP: {local_ip}")
        return local_ip
    except socket.error:
        logging.warning("Failed to get local IP, defaulting to 127.0.0.1")
        return "127.0.0.1"


def start_server_background():
    """Start the multiplayer server in the background."""
    try:
        import subprocess
        import sys
        import os
        server_path = os.path.join(os.path.dirname(__file__), 'multiplayer_server.py')
        if os.path.exists(server_path):
            subprocess.Popen([sys.executable, server_path])
            logging.info("Multiplayer server started in the background.")
            return True
        else:
            logging.error("multiplayer_server.py not found.")
            return False
    except Exception as e:
        logging.error(f"Failed to start multiplayer server: {e}")
        return False
