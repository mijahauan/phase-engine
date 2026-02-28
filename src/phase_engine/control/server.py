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
    def __init__(
        self,
        engine,
        channel_manager: VirtualChannelManager,
        status_address: str = "239.1.2.3",
        control_port: int = 5006,
    ):
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
        self._cmd_sock.bind(("", self.control_port))

        # Join the multicast group so we can receive commands sent to the status address
        import struct

        try:
            mreq = struct.pack(
                "=4s4s", socket.inet_aton(self.status_address), socket.inet_aton("0.0.0.0")
            )
            self._cmd_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            logger.info(f"Control server joined multicast group {self.status_address}")
        except OSError as e:
            logger.warning(f"Could not join multicast group {self.status_address}: {e}")

        # 2. Status Multicast Socket
        self._status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._status_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

        self._listener_thread = threading.Thread(target=self._command_listener_loop, daemon=True)
        self._status_thread = threading.Thread(target=self._status_multicaster_loop, daemon=True)

        self._listener_thread.start()
        self._status_thread.start()
        logger.info(
            f"Phase Engine Control Server listening on UDP {self.control_port}, multicasting status to {self.status_address}"
        )

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
                logger.debug(f"Control server received {len(data)} bytes from {addr}")
                if not data:
                    continue

                tlv = decode_tlv_packet(data)
                logger.debug(f"Decoded TLV: {tlv}")
                if tlv.get("_packet_type") == 1:  # CMD
                    self._handle_command(tlv, addr)

            except socket.error:
                if not self._running:
                    break
            except ValueError as e:
                logger.error(f"Error processing command values: {e}")
            except OSError as e:
                logger.error(f"Socket error processing command: {e}")

    def _handle_command(self, tlv: dict, addr: tuple):
        """Map TLV commands to engine actions."""
        ssrc = tlv.get(StatusType.OUTPUT_SSRC)
        logger.info(f"Control server received CMD: {tlv}")
        if not ssrc:
            return

        freq = tlv.get(StatusType.RADIO_FREQUENCY)
        preset = tlv.get(StatusType.PRESET)
        dest_sock = tlv.get(StatusType.OUTPUT_DATA_DEST_SOCKET)
        cmd_tag = tlv.get(StatusType.COMMAND_TAG)
        rate = tlv.get(StatusType.OUTPUT_SAMPRATE)

        # Update the virtual channel configuration
        params = {}
        if freq is not None:
            params["frequency_hz"] = freq
        if preset is not None:
            params["preset"] = preset
        if dest_sock is not None:
            params["destination"] = dest_sock
        if rate is not None:
            params["sample_rate"] = rate

        if params:
            self.channel_manager.configure_channel(ssrc, params)

        # Send STATUS ACK back to the requester
        self._send_status_ack(ssrc, cmd_tag, addr)

    def _send_status_ack(self, ssrc: int, cmd_tag: int, addr: tuple):
        """Send a basic TLV status packet back to acknowledge the command."""
        import struct

        resp = bytearray([0])  # STATUS packet

        if cmd_tag is not None:
            resp.extend(bytes([StatusType.COMMAND_TAG, 4]))
            resp.extend(struct.pack(">I", cmd_tag))

        resp.extend(bytes([StatusType.OUTPUT_SSRC, 4]))
        resp.extend(struct.pack(">I", ssrc))

        resp.extend(bytes([StatusType.EOL]))

        try:
            self._cmd_sock.sendto(resp, addr)
        except OSError as e:
            logger.debug(f"Failed to send ACK: {e}")

    def _status_multicaster_loop(self):
        """Periodically broadcast binary TLV status so ka9q tools can discover us."""
        import struct

        while self._running:
            try:
                for chan in self.channel_manager.get_channels():
                    buf = bytearray()
                    buf.append(0)  # Packet Type: STATUS

                    # 18: OUTPUT_SSRC
                    ssrc = chan.get("ssrc", 0)
                    ssrc_bytes = ssrc.to_bytes(4, byteorder="big").lstrip(b"\x00")
                    if not ssrc_bytes:
                        buf.extend([18, 0])
                    else:
                        buf.extend([18, len(ssrc_bytes)])
                        buf.extend(ssrc_bytes)

                    # 1: FREQ
                    freq = chan.get("frequency_hz", 10e6)
                    buf.extend([1, 8])
                    buf.extend(struct.pack(">d", freq))

                    # 2: PRESET
                    preset = chan.get("preset", "iq").encode("utf-8") + b"\x00"
                    buf.extend([2, len(preset)])
                    buf.extend(preset)

                    # 3: SAMPLE_RATE
                    sr = chan.get("sample_rate", self.engine.sample_rate)
                    if sr is None:
                        sr = self.engine.sample_rate
                    sr_bytes = sr.to_bytes(4, byteorder="big").lstrip(b"\x00")
                    if not sr_bytes:
                        buf.extend([3, 0])
                    else:
                        buf.extend([3, len(sr_bytes)])
                        buf.extend(sr_bytes)

                    # 17: DEST_SOCKET
                    dest = chan.get("destination", "127.0.0.1:5004")
                    ip = dest.split(":")[0] if ":" in dest else dest
                    port = int(dest.split(":")[1]) if ":" in dest else 5004
                    buf.extend([17, 8])
                    buf.extend(struct.pack(">H", socket.AF_INET))
                    buf.extend(struct.pack(">H", port))
                    buf.extend(socket.inet_aton(ip))

                    # 21: RTP_DEST string fallback
                    dest_bytes = dest.encode("utf-8") + b"\x00"
                    buf.extend([21, len(dest_bytes)])
                    buf.extend(dest_bytes)

                    buf.extend([0, 0])  # EOL

                    self._status_sock.sendto(buf, (self.status_address, 5006))

            except struct.error as e:
                logger.debug(f"Status multicast packing error: {e}")
            except OSError as e:
                logger.debug(f"Status multicast socket error: {e}")

            time.sleep(0.5)
