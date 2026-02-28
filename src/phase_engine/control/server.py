"""
Control Plane Server. Maps radiod network interactions to the Phase Engine.
"""

import socket
import json
import logging
import threading
import time
from typing import Dict, Any, Optional

from .tlv import decode_tlv_packet, StatusType
from ..virtual_channel import VirtualChannelManager

logger = logging.getLogger(__name__)

class ControlServer:
    def __init__(self, engine, channel_manager: VirtualChannelManager, status_address: str = "239.1.2.3", control_port: int = 5006):
        self.engine = engine
        self.channel_manager = channel_manager
        self.status_address = status_address
        self.control_port = control_port
        
        self._running = False
        self._cmd_sock = None
        self._status_sock = None
        
        self._listener_thread = None
        self._status_thread = None
        
    def start(self):
        """Start the control server threads."""
        self._running = True
        
        # 1. Command Listener Socket
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._cmd_sock.bind(('0.0.0.0', self.control_port))
        
        # 2. Status Multicast Socket
        self._status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._status_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        
        self._listener_thread = threading.Thread(target=self._command_listener_loop, daemon=True)
        self._status_thread = threading.Thread(target=self._status_multicaster_loop, daemon=True)
        
        self._listener_thread.start()
        self._status_thread.start()
        logger.info(f"Phase Engine Control Server listening on UDP {self.control_port}, multicasting status to {self.status_address}")

    def stop(self):
        """Stop the control server."""
        self._running = False
        if self._cmd_sock:
            self._cmd_sock.close()
        if self._status_sock:
            self._status_sock.close()
        
    def _command_listener_loop(self):
        while self._running:
            try:
                data, addr = self._cmd_sock.recvfrom(4096)
                if not data:
                    continue
                
                tlv = decode_tlv_packet(data)
                if tlv.get('_packet_type') == 1:  # CMD
                    self._handle_command(tlv, addr)
                    
            except socket.error:
                if not self._running:
                    break
            except Exception as e:
                logger.error(f"Error processing command: {e}")

    def _handle_command(self, tlv: dict, addr: tuple):
        """Map TLV commands to engine actions."""
        ssrc = tlv.get(StatusType.OUTPUT_SSRC)
        if not ssrc:
            return
            
        freq = tlv.get(StatusType.RADIO_FREQUENCY)
        preset = tlv.get(StatusType.PRESET)
        dest_sock = tlv.get(StatusType.OUTPUT_DATA_DEST_SOCKET)
        cmd_tag = tlv.get(StatusType.COMMAND_TAG)
        
        # Update the virtual channel configuration
        params = {}
        if freq is not None:
            params["frequency_hz"] = freq
        if preset is not None:
            params["preset"] = preset
        if dest_sock is not None:
            params["destination"] = dest_sock
            
        if params:
            self.channel_manager.configure_channel(ssrc, params)
        
        # Send STATUS ACK back to the requester
        self._send_status_ack(ssrc, cmd_tag, addr)

    def _send_status_ack(self, ssrc: int, cmd_tag: int, addr: tuple):
        """Send a basic TLV status packet back to acknowledge the command."""
        import struct
        resp = bytearray([0]) # STATUS packet
        
        if cmd_tag is not None:
            resp.extend(bytes([StatusType.COMMAND_TAG, 4]))
            resp.extend(struct.pack('>I', cmd_tag))
            
        resp.extend(bytes([StatusType.OUTPUT_SSRC, 4]))
        resp.extend(struct.pack('>I', ssrc))
        
        resp.extend(bytes([StatusType.EOL]))
        
        try:
            self._cmd_sock.sendto(resp, addr)
        except Exception as e:
            logger.debug(f"Failed to send ACK: {e}")

    def _status_multicaster_loop(self):
        """Periodically broadcast JSON status so ka9q tools can discover us."""
        while self._running:
            try:
                # Format exactly as radiod native JSON status
                status_obj = {
                    "samprate": self.engine.sample_rate,
                    "channels": []
                }
                
                # Report virtual channels
                for chan in self.channel_manager.get_channels():
                    chan_info = {"ssrc": chan["ssrc"]}
                    if "frequency_hz" in chan:
                        chan_info["freq"] = chan["frequency_hz"]
                    if "preset" in chan:
                        chan_info["preset"] = chan["preset"]
                    if "destination" in chan:
                        dest_parts = chan["destination"].split(':')
                        chan_info["dest"] = dest_parts[0]
                        if len(dest_parts) > 1:
                            chan_info["port"] = int(dest_parts[1])
                    status_obj["channels"].append(chan_info)
                
                msg = json.dumps(status_obj).encode('utf-8')
                self._status_sock.sendto(msg, (self.status_address, 5006))
            except Exception as e:
                logger.debug(f"Status multicast error: {e}")
                
            time.sleep(1.0)
