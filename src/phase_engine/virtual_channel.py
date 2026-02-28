"""
Virtual Channel Management for Phase Engine.

Bridges the Control Proxy (which receives commands for specific SSRCs/Frequencies)
with the RTP Egress and the core Phase Combiner, keeping state in sync.
"""

import logging
from typing import Dict, Any, Optional
import threading

from .sources import BROADCASTS
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

    def _evaluate_channel(self, ssrc: int):
        """Check if channel is fully configured and needs a streamer spun up."""
        with self._lock:
            chan = self._channels[ssrc]

            # We need at least frequency and destination to start streaming
            if "frequency_hz" not in chan or "destination" not in chan:
                return

            if ssrc in self._streamers:
                return  # Already streaming

            freq = chan["frequency_hz"]
            dest = chan["destination"]

            # Parse IP/Port
            parts = dest.split(":")
            ip = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 5004

            # Find matching broadcast(s) for this frequency
            matching_broadcasts = [b for b in BROADCASTS if b.frequency_hz == freq]
            if not matching_broadcasts:
                logger.warning(f"Requested frequency {freq} not in known broadcasts.")
                return

            # Select the target based on 'target' parameter if provided, else use first
            target_name = chan.get("target")
            selected_bcast = matching_broadcasts[0]

            if target_name:
                for b in matching_broadcasts:
                    if b.call_sign == target_name:
                        selected_bcast = b
                        break

            # Store the selected broadcast on the channel config
            chan["broadcast"] = selected_bcast

            logger.info(
                f"Starting virtual stream for SSRC {ssrc}: {selected_bcast.call_sign} at {freq} Hz -> {ip}:{port}"
            )

            # Create streamer
            streamer = RtpStreamer(
                destination_ip=ip, port=port, ssrc=ssrc, sample_rate=self.engine.sample_rate
            )
            self._streamers[ssrc] = streamer

    def remove_channel(self, ssrc: int):
        with self._lock:
            if ssrc in self._streamers:
                self._streamers[ssrc].close()
                del self._streamers[ssrc]
            if ssrc in self._channels:
                del self._channels[ssrc]

    def get_channels(self) -> list:
        with self._lock:
            return list(self._channels.values())

    def get_streamer(self, ssrc: int) -> Optional[RtpStreamer]:
        with self._lock:
            return self._streamers.get(ssrc)

    def get_active_broadcasts(self) -> Dict[int, Dict[str, Any]]:
        """Returns map of SSRC -> active Virtual Channel Dict"""
        with self._lock:
            return {
                ssrc: chan
                for ssrc, chan in self._channels.items()
                if ssrc in self._streamers and "broadcast" in chan
            }
