"""Simple systemd sd_notify implementation in pure Python."""
import os
import socket

def sd_notify(state: str) -> bool:
    """
    Send state to systemd via the socket named in NOTIFY_SOCKET.
    
    Args:
        state: State string to send (e.g. "WATCHDOG=1" or "READY=1")
        
    Returns:
        True if notification was sent, False if not running under systemd
    """
    notify_socket = os.environ.get('NOTIFY_SOCKET')
    if not notify_socket:
        return False
        
    if notify_socket.startswith('@'):
        notify_socket = '\0' + notify_socket[1:]
        
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(state.encode(), notify_socket)
            return True
    except Exception:
        return False
