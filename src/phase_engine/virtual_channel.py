"""
Virtual Channel Management for Phase Engine.

Bridges the Control Proxy (which receives commands for specific SSRCs/Frequencies)
with the RTP Egress and the core Phase Combiner, keeping state in sync.

Channel lifecycle:
  1. Client sends TLV CMD with {SSRC, FREQ, PRESET, SAMPLE_RATE} — no destination.
  2. configure_channel() is called; _evaluate_channel() fires once freq+ssrc are known.
  3. engine.open_channel(freq) lazily creates physical channels on all radiod sources.
  4. A deterministic output multicast address is assigned for the combined RTP stream.
  5. The address is advertised via the control server's status multicast so
     ka9q-python's discover_channels() can find it and subscribe.
"""

import hashlib
import logging
from typing import Dict, Any, Optional
import threading

from .data.rtp_packetizer import RtpStreamer

logger = logging.getLogger(__name__)


class VirtualChannelManager:
    """Manages the lifecycle of virtual channels served by the engine."""

    def __init__(self, engine):
        self.engine = engine
        self._channels: Dict[int, Dict[str, Any]] = {}
        self._streamers: Dict[int, RtpStreamer] = {}
        self._lock = threading.Lock()

    def allocate_channel(self, ssrc: int):
        with self._lock:
            if ssrc not in self._channels:
                self._channels[ssrc] = {"ssrc": ssrc}

    def configure_channel(self, ssrc: int, params: Dict[str, Any]):
        with self._lock:
            if ssrc not in self._channels:
                self._channels[ssrc] = {"ssrc": ssrc}

            for k, v in params.items():
                self._channels[ssrc][k] = v

        self._evaluate_channel(ssrc)

    @staticmethod
    def _assign_output_address(ssrc: int) -> tuple:
        """
        Assign a deterministic output multicast address for this SSRC.

        Uses the SSRC to derive a stable address in the 239.x.x.x
        administratively-scoped block so that the same SSRC always
        maps to the same multicast group.  Clients discover this via
        the status multicast; they never supply it themselves.

        Returns:
            (ip_str, port) e.g. ('239.76.21.3', 5004)
        """
        h = hashlib.sha256(ssrc.to_bytes(4, "big")).digest()
        # Use bytes 1-3 of the hash for the last three octets (239.x.x.x)
        ip = f"239.{h[0] & 0x7F}.{h[1]}.{h[2]}"
        port = 5004
        return ip, port

    def _evaluate_channel(self, ssrc: int):
        """Check if channel is fully configured and needs a streamer spun up."""
        with self._lock:
            chan = self._channels[ssrc]

            # We need at least a frequency to start — destination is assigned by us
            if "frequency_hz" not in chan:
                return

            if ssrc in self._streamers:
                return  # Already streaming

            freq = chan["frequency_hz"]

            # Assign a deterministic output multicast address for the combined stream.
            # This is what we advertise back in the status multicast so clients can
            # subscribe.  Clients (hf-timestd / ka9q-python) never send a destination.
            ip, port = self._assign_output_address(ssrc)
            chan["destination"] = f"{ip}:{port}"

        # Open physical channels on all radiod sources at this frequency.
        # Done outside the lock because engine.open_channel() acquires its own.
        try:
            self.engine.open_channel(freq)
        except (RuntimeError, OSError) as e:
            logger.error(f"Failed to open physical channels for SSRC {ssrc} at {freq} Hz: {e}")
            return

        with self._lock:
            chan = self._channels[ssrc]
            # Retrieve the destination that was just assigned
            ip, port = chan["destination"].split(":")
            port = int(port)

            logger.info(
                f"Starting virtual stream for SSRC {ssrc} at {freq} Hz -> {ip}:{port}"
            )

            # Create streamer targeting our assigned multicast address
            streamer = RtpStreamer(
                destination_ip=ip, port=port, ssrc=ssrc, sample_rate=self.engine.sample_rate
            )
            self._streamers[ssrc] = streamer

    def remove_channel(self, ssrc: int):
        freq = None
        with self._lock:
            if ssrc in self._streamers:
                self._streamers[ssrc].close()
                del self._streamers[ssrc]
            if ssrc in self._channels:
                freq = self._channels[ssrc].get("frequency_hz")
                del self._channels[ssrc]

        if freq is not None:
            try:
                self.engine.close_channel(freq)
            except (RuntimeError, OSError) as e:
                logger.warning(f"Failed to close physical channels for SSRC {ssrc} at {freq} Hz: {e}")

    def get_channels(self) -> list:
        with self._lock:
            return list(self._channels.values())

    def get_streamer(self, ssrc: int) -> Optional[RtpStreamer]:
        with self._lock:
            return self._streamers.get(ssrc)

    def get_active_broadcasts(self) -> Dict[int, Dict[str, Any]]:
        """Returns map of SSRC -> active Virtual Channel Dict (has streamer + frequency)."""
        with self._lock:
            return {
                ssrc: chan
                for ssrc, chan in self._channels.items()
                if ssrc in self._streamers and "frequency_hz" in chan
            }
