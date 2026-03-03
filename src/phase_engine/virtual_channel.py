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
import json
import logging
import os
from typing import Dict, Any, Optional
import threading

from .data.rtp_packetizer import RtpStreamer

logger = logging.getLogger(__name__)


class VirtualChannelManager:
    """Manages the lifecycle of virtual channels served by the engine."""

    # Persistent state file for surviving phase-engine restarts.
    STATE_FILE = "/var/lib/phase-engine/channels.json"

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
        self._save_state()

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
        """Check if channel is fully configured and needs a streamer spun up.

        Assigns the output destination immediately (so the status multicast
        advertises the channel right away) then provisions radiod sources and
        spins up the RTP streamer in a background thread so the control-server
        command listener is never blocked.
        """
        with self._lock:
            chan = self._channels[ssrc]

            # We need at least a frequency to start — destination is assigned by us
            if "frequency_hz" not in chan:
                return

            if ssrc in self._streamers:
                return  # Already streaming

            if chan.get("_provisioning"):
                return  # Already being provisioned in background

            freq = chan["frequency_hz"]

            # Assign a deterministic output multicast address for the combined stream.
            # This is what we advertise back in the status multicast so clients can
            # subscribe.  Clients (hf-timestd / ka9q-python) never send a destination.
            ip, port = self._assign_output_address(ssrc)
            chan["destination"] = f"{ip}:{port}"
            chan["_provisioning"] = True

        # Run the slow radiod provisioning + streamer creation in a background
        # thread so that the command listener can continue processing CMDs and
        # answering discovery polls while this channel is being set up.
        t = threading.Thread(
            target=self._provision_channel,
            args=(ssrc, freq),
            name=f"Provision-{ssrc}",
            daemon=True,
        )
        t.start()

    def _provision_channel(self, ssrc: int, freq: float):
        """Background worker: open physical radiod channels and start RTP streamer."""
        try:
            with self._lock:
                chan = self._channels.get(ssrc)
                if chan is None:
                    return
                params = {
                    'sample_rate': chan.get('sample_rate'),
                    'preset': chan.get('preset'),
                }

            self.engine.open_channel(freq, **params)
        except (RuntimeError, OSError) as e:
            logger.error(f"Failed to open physical channels for SSRC {ssrc} at {freq} Hz: {e}")
            with self._lock:
                chan = self._channels.get(ssrc)
                if chan:
                    chan.pop("_provisioning", None)
            return

        with self._lock:
            chan = self._channels.get(ssrc)
            if chan is None:
                return
            chan.pop("_provisioning", None)

            ip, port = chan["destination"].split(":")
            port = int(port)

            logger.info(
                f"Starting virtual stream for SSRC {ssrc} at {freq} Hz -> {ip}:{port}"
            )

            sample_rate = chan.get("sample_rate", self.engine.sample_rate)
            streamer = RtpStreamer(
                destination_ip=ip, port=port, ssrc=ssrc, sample_rate=sample_rate
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

        self._save_state()

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

    # ------------------------------------------------------------------
    # Persistent channel state
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist current channel definitions to disk.

        Only serialisable keys are saved (ssrc, frequency_hz, sample_rate,
        preset, encoding, combining_method, bearing_deg).  Runtime artefacts
        like destination and streamer references are excluded.
        """
        persist_keys = {
            "ssrc", "frequency_hz", "sample_rate", "preset",
            "encoding", "combining_method", "bearing_deg", "demod_type",
        }
        with self._lock:
            records = []
            for chan in self._channels.values():
                record = {k: v for k, v in chan.items() if k in persist_keys}
                if "ssrc" in record and "frequency_hz" in record:
                    records.append(record)

        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            tmp = self.STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(records, f, indent=2)
            os.replace(tmp, self.STATE_FILE)
            logger.debug(f"Saved {len(records)} channel(s) to {self.STATE_FILE}")
        except OSError as e:
            logger.warning(f"Failed to persist channel state: {e}")

    def restore_channels(self) -> int:
        """Restore channels from disk after a phase-engine restart.

        Replays every saved channel through configure_channel() so that
        physical radiod channels are re-created and egress streamers spun
        up — without waiting for the client to re-request them.

        Returns:
            Number of channels restored.
        """
        if not os.path.exists(self.STATE_FILE):
            logger.info("No saved channel state found — starting fresh")
            return 0

        try:
            with open(self.STATE_FILE, "r") as f:
                records = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load channel state: {e}")
            return 0

        restored = 0
        for rec in records:
            ssrc = rec.get("ssrc")
            if ssrc is None:
                continue
            params = {k: v for k, v in rec.items() if k != "ssrc"}
            try:
                self.configure_channel(ssrc, params)
                restored += 1
                logger.info(
                    f"Restored channel SSRC {ssrc} at "
                    f"{params.get('frequency_hz', 0) / 1e6:.3f} MHz"
                )
            except Exception as e:
                logger.error(f"Failed to restore channel SSRC {ssrc}: {e}")

        logger.info(f"Restored {restored}/{len(records)} channel(s) from {self.STATE_FILE}")
        return restored
