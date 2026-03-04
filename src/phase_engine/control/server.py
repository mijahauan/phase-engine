"""
Control Plane Server — maps ka9q-radio network interactions to the Phase Engine.

Implements two threads:

1. **Command listener** — receives UDP TLV CMD packets on the status multicast
   group (port 5006).  Handles channel creation, discovery polls
   (``SSRC=0xFFFFFFFF``), and echoes STATUS ACKs with the engine-assigned output
   multicast address.

2. **Status multicaster** — periodically broadcasts binary TLV STATUS packets
   (every 0.5 s) so that ``ka9q-python``'s ``discover_channels()`` can find
   phase-engine channels exactly as it would find native radiod channels.

Together these two threads make phase-engine indistinguishable from a real
radiod instance on the network.
"""

import struct
import socket
import json
import logging
import threading
import time
from typing import Dict, Any, List, Optional

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
        self._timing_thread = None

        # Cached upstream timing metadata forwarded in our status broadcasts.
        # Protected by _timing_lock; updated by _timing_poller_loop.
        self._timing_lock = threading.Lock()
        self._gps_time: Optional[int] = None      # nanoseconds since Unix epoch
        self._rtp_timesnap: Optional[int] = None   # 32-bit RTP counter snapshot

    def start(self):
        """Start the control server threads."""
        self._running = True

        # 1. Command Listener Socket
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._cmd_sock.bind((self.status_address, self.control_port))

        # Join the multicast group so we can receive commands sent to the status address
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
        self._timing_thread = threading.Thread(target=self._timing_poller_loop, daemon=True, name="TimingPoller")

        self._listener_thread.start()
        self._status_thread.start()
        self._timing_thread.start()
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

        if not ssrc:
            return

        # SSRC 0xFFFFFFFF is a broadcast discovery probe from ka9q tools.
        # Respond by immediately broadcasting current channel status so that
        # discover_channels() sees our channels (not stale radiod channels
        # leaking from other multicast groups on the LAN).
        if ssrc == 0xFFFFFFFF:
            self._broadcast_status_now()
            return

        logger.info(f"Control server received CMD: {tlv}")

        freq = tlv.get(StatusType.RADIO_FREQUENCY)
        preset = tlv.get(StatusType.PRESET)
        if preset:
            preset = preset.rstrip('\x00').strip()
        cmd_tag = tlv.get(StatusType.COMMAND_TAG)
        rate = tlv.get(StatusType.OUTPUT_SAMPRATE)
        demod_type = tlv.get(StatusType.DEMOD_TYPE)
        encoding = tlv.get(StatusType.OUTPUT_ENCODING)

        # NOTE: clients (hf-timestd via ka9q-python) do NOT send a destination.
        # We never accept OUTPUT_DATA_DEST_SOCKET from the client; instead we assign
        # our own deterministic output multicast address inside configure_channel().
        params = {}
        if freq is not None:
            params["frequency_hz"] = freq
        if preset is not None:
            params["preset"] = preset
        if rate is not None:
            params["sample_rate"] = rate
        if demod_type is not None:
            params["demod_type"] = demod_type
        if encoding is not None:
            params["encoding"] = encoding

        if params:
            self.channel_manager.configure_channel(ssrc, params)

        # Retrieve the assigned destination so we can echo it back in the ACK.
        # After configure_channel() the channel dict will have "destination" set.
        assigned_dest = None
        channels = self.channel_manager.get_channels()
        for chan in channels:
            if chan.get("ssrc") == ssrc:
                assigned_dest = chan.get("destination")
                break

        # Send STATUS ACK back to the requester (includes our assigned multicast addr)
        self._send_status_ack(ssrc, cmd_tag, addr, assigned_dest)

    def _send_status_ack(self, ssrc: int, cmd_tag: int, addr: tuple, assigned_dest: str = None):
        """
        Send a TLV status packet back to acknowledge the command.

        Includes the engine-assigned output multicast address (OUTPUT_DATA_DEST_SOCKET)
        so that ka9q-python's ensure_channel() / discover_channels() can immediately
        learn where to subscribe for the combined RTP stream.
        """
        resp = bytearray([0])  # STATUS packet

        if cmd_tag is not None:
            resp.extend(bytes([StatusType.COMMAND_TAG, 4]))
            resp.extend(struct.pack(">I", cmd_tag & 0xFFFFFFFF))

        resp.extend(bytes([StatusType.OUTPUT_SSRC, 4]))
        resp.extend(struct.pack(">I", ssrc & 0xFFFFFFFF))

        # Encode our assigned output multicast address so the client knows where
        # to subscribe.  Format: AF_INET (2 bytes) + port (2 bytes) + IPv4 (4 bytes)
        if assigned_dest:
            try:
                dest_parts = assigned_dest.split(":")
                dest_ip = dest_parts[0]
                dest_port = int(dest_parts[1]) if len(dest_parts) > 1 else 5004
                resp.extend(bytes([StatusType.OUTPUT_DATA_DEST_SOCKET, 8]))
                resp.extend(struct.pack(">H", socket.AF_INET))
                resp.extend(struct.pack(">H", dest_port))
                resp.extend(socket.inet_aton(dest_ip))
            except (OSError, ValueError) as e:
                logger.debug(f"Could not encode destination in ACK: {e}")

        resp.extend(bytes([StatusType.EOL]))

        try:
            self._cmd_sock.sendto(resp, addr)
        except OSError as e:
            logger.debug(f"Failed to send ACK: {e}")

    def _broadcast_status_now(self):
        """Broadcast current channel status immediately (used for poll responses)."""
        # Snapshot the cached upstream timing under lock once per broadcast round
        with self._timing_lock:
            gps_time = self._gps_time
            rtp_timesnap = self._rtp_timesnap

        try:
            for chan in self.channel_manager.get_channels():
                ssrc = chan.get("ssrc")
                freq = chan.get("frequency_hz")
                dest = chan.get("destination")
                if not ssrc or not freq or not dest:
                    continue

                buf = bytearray()
                buf.append(0)  # Packet Type: STATUS

                # OUTPUT_SSRC
                buf.extend([StatusType.OUTPUT_SSRC, 4])
                buf.extend(struct.pack(">I", ssrc & 0xFFFFFFFF))

                # GPS_TIME — forwarded from upstream radiod
                if gps_time is not None:
                    buf.extend([StatusType.GPS_TIME, 8])
                    buf.extend(struct.pack(">q", gps_time))

                # RTP_TIMESNAP — forwarded from upstream radiod
                if rtp_timesnap is not None:
                    buf.extend([StatusType.RTP_TIMESNAP, 4])
                    buf.extend(struct.pack(">I", rtp_timesnap & 0xFFFFFFFF))

                # RADIO_FREQUENCY
                buf.extend([StatusType.RADIO_FREQUENCY, 8])
                buf.extend(struct.pack(">d", freq))

                # PRESET
                preset_bytes = chan.get("preset", "iq").encode("utf-8")
                buf.extend([StatusType.PRESET, len(preset_bytes)])
                buf.extend(preset_bytes)

                # OUTPUT_SAMPRATE
                sr = chan.get("sample_rate") or self.engine.sample_rate
                buf.extend([StatusType.OUTPUT_SAMPRATE, 4])
                buf.extend(struct.pack(">I", sr))

                # OUTPUT_DATA_DEST_SOCKET — the engine-assigned output multicast address.
                # This is what ka9q-python's discover_channels() reads to subscribe.
                dest_ip = dest.split(":")[0] if ":" in dest else dest
                dest_port = int(dest.split(":")[1]) if ":" in dest else 5004
                buf.extend([StatusType.OUTPUT_DATA_DEST_SOCKET, 8])
                buf.extend(struct.pack(">H", socket.AF_INET))
                buf.extend(struct.pack(">H", dest_port))
                buf.extend(socket.inet_aton(dest_ip))

                # OUTPUT_ENCODING
                enc = chan.get("encoding", 0)
                if enc:
                    buf.extend([StatusType.OUTPUT_ENCODING, 1])
                    buf.extend([enc & 0xFF])

                buf.extend([StatusType.EOL, 0])

                self._status_sock.sendto(buf, (self.status_address, 5006))

        except struct.error as e:
            logger.debug(f"Status multicast packing error: {e}")
        except OSError as e:
            logger.debug(f"Status multicast socket error: {e}")

    def _status_multicaster_loop(self):
        """Periodically broadcast binary TLV status so ka9q tools can discover us."""
        while self._running:
            self._broadcast_status_now()
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # Upstream timing poller
    # ------------------------------------------------------------------

    def _get_upstream_status_addresses(self) -> List[str]:
        """Return the status addresses of the engine's upstream radiod sources."""
        addrs = []
        for source in self.engine.sources.values():
            if hasattr(source, "status_address") and source.status_address:
                addrs.append(source.status_address)
        return addrs

    def _timing_poller_loop(self):
        """Periodically poll upstream radiod sources for GPS_TIME / RTP_TIMESNAP.

        The values are cached under ``_timing_lock`` and included in every
        subsequent status broadcast so that downstream consumers (hf-timestd)
        can correlate RTP timestamps with UTC.
        """
        from ka9q import discover_channels

        logger.info("Timing poller started")

        # Wait briefly for the engine to finish connecting before polling
        time.sleep(2.0)

        while self._running:
            try:
                addrs = self._get_upstream_status_addresses()
                if not addrs:
                    time.sleep(5.0)
                    continue

                # Poll the first (reference) source — all radiod instances
                # share the same GPSDO so any source will have valid timing.
                ref_addr = addrs[0]
                channels = discover_channels(ref_addr, listen_duration=1.5)

                best_gps = None
                best_rtp = None
                for _ssrc, info in channels.items():
                    g = getattr(info, "gps_time", None)
                    r = getattr(info, "rtp_timesnap", None)
                    if g is not None and r is not None:
                        best_gps = g
                        best_rtp = r
                        break  # One valid pair is sufficient

                if best_gps is not None and best_rtp is not None:
                    with self._timing_lock:
                        changed = (self._gps_time != best_gps or
                                   self._rtp_timesnap != best_rtp)
                        self._gps_time = best_gps
                        self._rtp_timesnap = best_rtp
                    if changed:
                        logger.info(
                            f"Upstream timing updated: GPS_TIME={best_gps}, "
                            f"RTP_TIMESNAP={best_rtp}"
                        )
                else:
                    logger.debug(
                        f"No GPS_TIME/RTP_TIMESNAP from upstream {ref_addr}"
                    )

            except Exception as e:
                logger.warning(f"Timing poller error: {e}")

            time.sleep(2.0)
