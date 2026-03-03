"""
RadiodSource - Manages a single radiod instance as a sample source.

Each RadiodSource wraps a RadiodControl and manages channels/streams
independently. Multiple RadiodSources can be combined by the phase-engine
for coherent array processing.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
import threading
import time
import logging

from ka9q import RadiodControl, RTPRecorder, Encoding, discover_channels

logger = logging.getLogger(__name__)


@dataclass
class ChannelConfig:
    """Configuration for a single channel."""

    frequency_hz: float
    sample_rate: int = 12000
    preset: str = "iq"
    encoding: int = 4  # F32


@dataclass
class ChannelState:
    """Runtime state for an active channel."""

    ssrc: int
    frequency_hz: float
    multicast_address: str
    port: int
    recorder: Optional[RTPRecorder] = None
    samples: List[np.ndarray] = field(default_factory=list)
    first_timestamp: Optional[int] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class RadiodSource:
    """
    Manages a single radiod instance as a sample source for the phased array.

    Each RadiodSource:
    - Maintains its own RadiodControl connection
    - Creates channels with unique SSRCs (within this radiod)
    - Captures samples via RTPRecorder
    - Provides sample buffers for alignment and combining

    Usage:
        source = RadiodSource("bee1", "bee1-status.local")
        source.connect()
        source.create_channel(900e3)  # 900 kHz
        source.start_capture()
        # ... later ...
        samples = source.get_samples(900e3, duration_sec=2.0)
        source.stop_capture()
        source.disconnect()
    """

    def __init__(
        self,
        name: str,
        status_address: str,
        ssrc_base: int = None,
        default_sample_rate: int = 12000,
        default_encoding: int = 4,  # F32
    ):
        """
        Initialize a RadiodSource.

        Args:
            name: Human-readable name for this source (e.g., "bee1")
            status_address: radiod status multicast address (e.g., "bee1-status.local")
            ssrc_base: Base SSRC for channels created by this source.
                       If None, derived from a hash of the source name so each
                       source gets a unique, stable SSRC range.
            default_sample_rate: Default sample rate for new channels
            default_encoding: Default encoding (4 = F32)
        """
        self.name = name
        self.status_address = status_address
        # Derive a unique base from the source name so that multiple RadiodSources
        # never collide on the same SSRC even when they all start at counter=0.
        if ssrc_base is None:
            h = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
            ssrc_base = (h & 0x7FFF0000) | 0x00010000  # upper 15 bits of hash, min 65536
        self.ssrc_base = ssrc_base
        self.default_sample_rate = default_sample_rate
        self.default_encoding = default_encoding

        self._control: Optional[RadiodControl] = None
        self._channels: Dict[float, ChannelState] = {}  # freq_hz -> state
        self._ssrc_counter = 0
        self._lock = threading.Lock()
        self._capturing = False
        self._health_thread = None
        self._last_packet_times = {}  # freq_hz -> timestamp

    def connect(self) -> None:
        """Connect to the radiod instance."""
        if self._control is not None:
            return

        logger.info(f"{self.name}: Connecting to {self.status_address}")
        self._control = RadiodControl(self.status_address)
        logger.info(f"{self.name}: Connected")

    def disconnect(self) -> None:
        """Disconnect from the radiod instance."""
        if self._control is None:
            return

        self.stop_capture()

        # Remove all channels
        for freq_hz, state in list(self._channels.items()):
            try:
                self._control.remove_channel(state.ssrc)
                logger.debug(f"{self.name}: Removed channel {freq_hz/1e6:.3f} MHz")
            except OSError as e:
                logger.warning(f"{self.name}: Network failed to remove channel: {e}")
            except ValueError as e:
                logger.warning(f"{self.name}: Value error removing channel: {e}")

        self._channels.clear()
        self._control.close()
        self._control = None
        logger.info(f"{self.name}: Disconnected")

    def create_channel(
        self,
        frequency_hz: float,
        sample_rate: Optional[int] = None,
        preset: str = "iq",
    ) -> int:
        """
        Create a channel at the specified frequency.

        Args:
            frequency_hz: Center frequency in Hz
            sample_rate: Sample rate (default: self.default_sample_rate)
            preset: Channel preset (default: "iq")

        Returns:
            SSRC of the created channel
        """
        if self._control is None:
            raise RuntimeError(f"{self.name}: Not connected")

        if frequency_hz in self._channels:
            return self._channels[frequency_hz].ssrc

        sample_rate = int(sample_rate or self.default_sample_rate)

        logger.debug(f"{self.name}: Ensuring channel at {frequency_hz/1e6:.3f} MHz")

        try:
            channel_info = self._control.ensure_channel(
                frequency_hz=frequency_hz,
                preset=preset,
                sample_rate=sample_rate,
                encoding=self.default_encoding,
                timeout=5.0
            )
        except Exception as e:
            logger.error(f"{self.name}: Failed to ensure channel at {frequency_hz/1e6:.3f} MHz: {e}")
            raise RuntimeError(f"ensure_channel failed: {e}")

        with self._lock:
            self._channels[frequency_hz] = ChannelState(
                ssrc=channel_info.ssrc,
                frequency_hz=frequency_hz,
                multicast_address=channel_info.multicast_address,
                port=channel_info.port,
            )

        return channel_info.ssrc

    def create_channels(self, frequencies_hz: List[float], **kwargs) -> Dict[float, int]:
        """
        Create multiple channels.

        Args:
            frequencies_hz: List of frequencies in Hz
            **kwargs: Passed to create_channel

        Returns:
            Dict mapping frequency_hz -> SSRC
        """
        result = {}
        for freq_hz in frequencies_hz:
            result[freq_hz] = self.create_channel(freq_hz, **kwargs)
        return result

    def remove_channel(self, frequency_hz: float) -> None:
        """Remove a channel."""
        if self._control is None:
            return

        state = self._channels.pop(frequency_hz, None)
        if state is None:
            return

        if state.recorder is not None:
            state.recorder.stop_recording()
            state.recorder.stop()

        self._control.remove_channel(state.ssrc)
        logger.debug(f"{self.name}: Removed channel at {frequency_hz/1e6:.3f} MHz")

    def start_capture(self, frequencies_hz: Optional[List[float]] = None) -> None:
        """
        Start capturing samples from channels.

        Args:
            frequencies_hz: List of frequencies to capture (default: all channels)
        """
        if self._capturing:
            return

        if frequencies_hz is None:
            frequencies_hz = list(self._channels.keys())

        for freq_hz in frequencies_hz:
            state = self._channels.get(freq_hz)
            if state is None:
                continue

            # Clear previous samples
            with state.lock:
                state.samples.clear()

            # Create packet handler
            def make_handler(s: ChannelState):
                def handler(header, payload, wallclock):
                    import time
                    self._last_packet_times[s.frequency_hz] = time.time()
                    
                    samples = np.frombuffer(payload, dtype=np.float32).view(np.complex64)
                    with s.lock:
                        if not s.samples:
                            s.first_timestamp = header.timestamp
                        s.samples.append(samples.copy())

                return handler

            from ka9q.discovery import ChannelInfo
            channel_info = ChannelInfo(
                ssrc=state.ssrc,
                frequency=state.frequency_hz,
                sample_rate=self.default_sample_rate,
                multicast_address=state.multicast_address,
                port=state.port,
                preset="iq",
                encoding=self.default_encoding,
                snr=0.0, # Dummy SNR
            )

            state.recorder = RTPRecorder(channel=channel_info, on_packet=make_handler(state))
            state.recorder.start()
            state.recorder.start_recording()

        self._capturing = True
        logger.info(f"{self.name}: Started capture on {len(frequencies_hz)} channels")
        
        # Start health monitor if not running
        if self._health_thread is None or not self._health_thread.is_alive():
            import threading
            self._health_thread = threading.Thread(target=self._health_monitor_loop, daemon=True, name=f"{self.name}-Health")
            self._health_thread.start()

    def stop_capture(self) -> None:
        """Stop capturing samples."""
        if not self._capturing:
            return

        for state in self._channels.values():
            if state.recorder is not None:
                state.recorder.stop_recording()
                state.recorder.stop()
                state.recorder = None
                
        if self._health_thread is not None:
            self._health_thread.join(timeout=2.0)
            self._health_thread = None

        self._capturing = False
        self._health_thread = None
        self._last_packet_times = {}  # freq_hz -> timestamp
        logger.info(f"{self.name}: Stopped capture")

    def get_samples(
        self,
        frequency_hz: float,
        max_samples: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Get captured samples for a frequency (non-destructive peek).

        Args:
            frequency_hz: Frequency to get samples for
            max_samples: Maximum number of samples to return (default: all)

        Returns:
            Complex sample array, or None if no samples
        """
        state = self._channels.get(frequency_hz)
        if state is None:
            return None

        with state.lock:
            if not state.samples:
                return None
            samples = np.concatenate(state.samples)

        if max_samples is not None and len(samples) > max_samples:
            samples = samples[:max_samples]

        return samples

    def consume_samples(
        self,
        frequency_hz: float,
        max_samples: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Atomically read and clear captured samples for a frequency.

        Use this in the egress loop to avoid reprocessing stale data on
        every iteration. After this call the buffer is empty until new
        RTP packets arrive.

        Args:
            frequency_hz: Frequency to consume samples for
            max_samples: Maximum number of samples to return

        Returns:
            Tuple of (complex sample array, RTP timestamp of first sample),
            or None if no samples available
        """
        state = self._channels.get(frequency_hz)
        if state is None:
            return None

        with state.lock:
            if not state.samples:
                return None
            
            raw = list(state.samples)
            timestamp = state.first_timestamp
            state.samples.clear()
            state.first_timestamp = None

        if not raw or timestamp is None:
            return None

        samples = np.concatenate(raw)
        
        # If we hit max_samples, keep the leftovers in the buffer
        # and advance their timestamp so we don't lose them
        if max_samples is not None and len(samples) > max_samples:
            leftovers = samples[max_samples:]
            samples = samples[:max_samples]
            
            with state.lock:
                # Put leftovers back at the front
                state.samples.insert(0, leftovers)
                # If new packets arrived while we were concatenating, they go after leftovers.
                # The timestamp for the first element is now the original + max_samples
                if state.first_timestamp is None:
                    # No new packets arrived, or we're the first thing
                    state.first_timestamp = (timestamp + max_samples) % (2**32)
                else:
                    # New packets arrived, but our leftovers are older, so they dictate the start
                    state.first_timestamp = (timestamp + max_samples) % (2**32)

        return samples, timestamp

    def clear_samples(self, frequency_hz: Optional[float] = None) -> None:
        """
        Clear captured samples.

        Args:
            frequency_hz: Frequency to clear (default: all frequencies)
        """
        if frequency_hz is not None:
            state = self._channels.get(frequency_hz)
            if state is not None:
                with state.lock:
                    state.samples.clear()
        else:
            for state in self._channels.values():
                with state.lock:
                    state.samples.clear()

    def get_channel_info(self, frequency_hz: float) -> Optional[ChannelState]:
        """Get channel state for a frequency."""
        return self._channels.get(frequency_hz)

    @property
    def frequencies(self) -> List[float]:
        """List of active channel frequencies."""
        return list(self._channels.keys())

    @property
    def is_connected(self) -> bool:
        """Whether connected to radiod."""
        return self._control is not None

    @property
    def is_capturing(self) -> bool:
        """Whether currently capturing samples."""
        return self._capturing

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"RadiodSource({self.name!r}, {self.status_address!r}, {status})"

    def _health_monitor_loop(self):
        """Monitor channel health and attempt recovery if RTP drops."""
        import time
        logger.info(f"{self.name}: Health monitor started")
        
        while self._capturing:
            time.sleep(1.0)
            
            now = time.time()
            needs_recovery = False
            
            with self._lock:
                # Check for channels that haven't received packets in 3 seconds
                for freq_hz, state in self._channels.items():
                    last_pkt = self._last_packet_times.get(freq_hz, now)
                    if now - last_pkt > 3.0:
                        logger.warning(f"{self.name}: No RTP packets for {freq_hz/1e6:.3f} MHz in {now-last_pkt:.1f}s")
                        needs_recovery = True
                        break
                        
            if needs_recovery:
                logger.warning(f"{self.name}: Initiating source recovery...")
                try:
                    # 1. Force reconnect control socket
                    if self._control is not None:
                        self._control.close()
                    self._control = RadiodControl(self.status_address)
                    
                    # 2. Recreate all channels
                    with self._lock:
                        frequencies = list(self._channels.keys())
                        
                    for freq_hz in frequencies:
                        state = self._channels.get(freq_hz)
                        if not state:
                            continue
                            
                        # Try to recreate physical channel on backend
                        try:
                            channel_info = self._control.ensure_channel(
                                frequency_hz=freq_hz,
                                preset="iq",
                                sample_rate=int(self.default_sample_rate),
                                encoding=self.default_encoding,
                            )
                            logger.info(f"{self.name}: Recreated channel at {freq_hz/1e6:.3f} MHz (SSRC={channel_info.ssrc})")
                            
                            # If multicast address changed, recreate the recorder
                            if (state.multicast_address != channel_info.multicast_address or 
                                state.port != channel_info.port):
                                logger.info(f"{self.name}: Destination changed {state.multicast_address}:{state.port} -> {channel_info.multicast_address}:{channel_info.port}. Recreating RTPRecorder.")
                                
                                if state.recorder is not None:
                                    state.recorder.stop_recording()
                                    state.recorder.stop()
                                
                                state.ssrc = channel_info.ssrc
                                state.multicast_address = channel_info.multicast_address
                                state.port = channel_info.port
                                
                                def make_handler(s: ChannelState):
                                    def handler(header, payload, wallclock):
                                        import time
                                        self._last_packet_times[s.frequency_hz] = time.time()
                                        samples = np.frombuffer(payload, dtype=np.float32).view(np.complex64)
                                        with s.lock:
                                            if not s.samples:
                                                s.first_timestamp = header.timestamp
                                            s.samples.append(samples.copy())
                                    return handler
                                
                                state.recorder = RTPRecorder(channel=channel_info, on_packet=make_handler(state))
                                state.recorder.start()
                                state.recorder.start_recording()

                            # Reset timeout counter so we give it a chance to connect
                            self._last_packet_times[freq_hz] = time.time()
                        except Exception as e:
                            logger.error(f"{self.name}: Failed to recreate channel {freq_hz/1e6:.3f} MHz: {e}")
                            
                except Exception as e:
                    logger.error(f"{self.name}: Recovery attempt failed: {e}")
