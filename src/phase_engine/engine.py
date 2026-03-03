"""
PhaseEngine - Main orchestrator for coherent phased array processing.

Coordinates:
1. Multiple RadiodSources (one per physical radiod)
2. Adaptive per-block phase combining for all broadcasts
3. Virtual radio interface for downstream apps (hf-timestd)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import logging
import threading

import numpy as np

from .sources import (
    RadiodSource,
    Broadcast,
    BROADCASTS,
    FREQUENCIES_HZ,
    get_broadcasts_for_frequency,
)
from .config_loader import get_engine_kwargs
from .dsp.array_geometry import AntennaArray
from .dsp.combiner import PhaseCombiner

logger = logging.getLogger(__name__)


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing from point 1 to point 2 in degrees."""
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    initial_bearing = math.atan2(y, x)
    return (math.degrees(initial_bearing) + 360) % 360


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for a radiod source."""

    name: str
    status_address: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z in meters
    enabled: bool = True


class PhaseEngine:
    """
    Main orchestrator for the phased array processing system.

    Lifecycle:
        1. Create PhaseEngine with QTH and source configs
        2. Call connect() to connect to all radiods
        3. Call start() to begin continuous processing
        4. Combined streams are available via get_combined_samples()
        5. Call stop() and disconnect() when done
    """

    def __init__(
        self,
        qth_latitude: float,
        qth_longitude: float,
        sources: List[SourceConfig],
        reference_source: Optional[str] = None,
        sample_rate: int = 12000,
        **kwargs,
    ):
        """
        Initialize the PhaseEngine.

        Args:
            qth_latitude: Observer latitude (degrees)
            qth_longitude: Observer longitude (degrees)
            sources: List of source configurations
            reference_source: Name of reference source (default: first source)
            sample_rate: Sample rate for all channels
        """
        self.qth_latitude = qth_latitude
        self.qth_longitude = qth_longitude
        self.sample_rate = sample_rate

        # Create RadiodSources
        self.sources: Dict[str, RadiodSource] = {}
        self.source_positions: Dict[str, Tuple[float, float, float]] = {}

        for i, config in enumerate(sources):
            if not config.enabled:
                continue

            self.sources[config.name] = RadiodSource(config.name, config.status_address)
            self.source_positions[config.name] = config.position

            if reference_source is None:
                reference_source = config.name

        self.reference_source = reference_source

        # Build DSP Array
        self.array = AntennaArray(self.reference_source, self.source_positions)
        self.combiner = PhaseCombiner(self.array)

        self._running = False
        self._lock = threading.Lock()

        logger.info(
            f"PhaseEngine initialized with {len(self.sources)} sources, "
            f"reference={self.reference_source}"
        )

    def connect(self) -> None:
        """Connect to all radiod sources."""
        logger.info("Connecting to radiod sources...")
        for source in self.sources.values():
            source.connect()
        logger.info(f"Connected to {len(self.sources)} sources")

    def disconnect(self) -> None:
        """Disconnect from all radiod sources."""
        self.stop()
        logger.info("Disconnecting from radiod sources...")
        for source in self.sources.values():
            source.disconnect()
        logger.info("Disconnected")

    def create_broadcast_channels(self) -> None:
        """
        Create channels for all 9 frequencies on all sources.

        This sets up the physical channels needed to receive
        all 17 broadcasts (9 frequencies, 4 shared).
        """
        logger.info(f"Creating channels for {len(FREQUENCIES_HZ)} frequencies...")

        for source in self.sources.values():
            source.create_channels(FREQUENCIES_HZ)

        logger.info(f"Created {len(FREQUENCIES_HZ)} channels on {len(self.sources)} sources")

    def open_channel(self, frequency_hz: float, **kwargs) -> None:
        """
        Lazily open a physical channel at the requested frequency on all sources.

        Called by VirtualChannelManager when a client (e.g. hf-timestd via
        ka9q-python) sends a TLV CMD requesting a frequency.  Creates the
        channel on every connected RadiodSource and starts capture so that
        combined samples are available for the egress loop.

        Args:
            frequency_hz: Center frequency in Hz
            **kwargs: Passed to source.create_channel (e.g., sample_rate, preset)
        """
        with self._lock:
            for name, source in self.sources.items():
                if not source.is_connected:
                    logger.warning(f"open_channel: source {name} not connected, skipping")
                    continue
                try:
                    if frequency_hz in source._channels:
                        logger.debug(f"open_channel: {name} already has {frequency_hz/1e6:.3f} MHz")
                        continue
                    
                    # Pass requested sample_rate or fallback to engine default
                    sample_rate = kwargs.get('sample_rate', self.sample_rate)
                    preset = kwargs.get('preset', 'iq')
                    
                    source.create_channel(frequency_hz, sample_rate=sample_rate, preset=preset)
                    source.start_capture([frequency_hz])
                    logger.info(f"open_channel: {name} opened {frequency_hz/1e6:.3f} MHz")
                except (RuntimeError, OSError) as e:
                    logger.error(f"open_channel: {name} failed at {frequency_hz/1e6:.3f} MHz: {e}")

    def close_channel(self, frequency_hz: float) -> None:
        """
        Close the physical channel at the given frequency on all sources.

        Called by VirtualChannelManager when a virtual channel is removed.

        Args:
            frequency_hz: Center frequency in Hz
        """
        with self._lock:
            for name, source in self.sources.items():
                if not source.is_connected:
                    continue
                try:
                    source.remove_channel(frequency_hz)
                    logger.info(f"close_channel: {name} closed {frequency_hz/1e6:.3f} MHz")
                except (RuntimeError, OSError) as e:
                    logger.warning(f"close_channel: {name} failed at {frequency_hz/1e6:.3f} MHz: {e}")

    def start(self) -> None:
        """Start the engine and wait for client requests."""
        if self._running:
            return

        self._running = True
        logger.info("PhaseEngine started (idle, waiting for client requests)")

    def stop(self) -> None:
        """Stop continuous capture."""
        if not self._running:
            return

        for source in self.sources.values():
            source.stop_capture()

        self._running = False
        logger.info("PhaseEngine stopped")

    def get_combined_samples(
        self,
        virtual_channel: dict,
        max_samples: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Get combined samples for a virtual channel.

        Args:
            virtual_channel: Virtual channel state dictionary
            max_samples: Maximum samples to return

        Returns:
            Tuple of (combined complex sample array, RTP timestamp), or None
        """
        freq_hz = virtual_channel.get("frequency_hz")
        if not freq_hz:
            return None

        # Collect samples from all sources at this frequency.
        # Use consume_samples (destructive read) so data is not reprocessed.
        samples = {}
        timestamp = None
        for name, source in self.sources.items():
            result = source.consume_samples(freq_hz, max_samples)
            if result is not None:
                s, ts = result
                samples[name] = s
                if timestamp is None:
                    timestamp = ts

        if not samples or timestamp is None:
            return None

        # Determine the combining method requested by the client
        method = virtual_channel.get("combining_method", "mrc")

        try:
            # If the client supplied an explicit bearing use it; otherwise default to 0.0.
            # Spatial steering methods (beamform/mvdr) require a valid bearing; without one
            # they fall back to MRC so combining still works even for unknown transmitters.
            bearing_deg = virtual_channel.get("bearing_deg", 0.0)

            a_target = None
            if method in ["beamform", "mvdr", "null", "focus_null", "adaptive", "focus"]:
                if method in ["focus", "focus_null"]:
                    method = "mvdr"
                a_target = self.array.get_steering_vector(freq_hz, bearing_deg)

            combined = self.combiner.process(samples, method=method, steering_vector=a_target)
            if combined is None:
                return None
            return (combined, timestamp)

        except (ValueError, np.linalg.LinAlgError) as e:
            logger.error(f"Combining failed for SSRC {virtual_channel.get('ssrc')} at {freq_hz/1e6:.3f} MHz: {e}")
            return None

    def get_all_combined_samples(
        self,
        max_samples: Optional[int] = None,
    ) -> Dict[Broadcast, np.ndarray]:
        """
        Get combined samples for all broadcasts.

        Args:
            max_samples: Maximum samples per broadcast

        Returns:
            Dict mapping Broadcast -> combined samples
        """
        # Collect samples from all sources at all frequencies
        frequency_samples: Dict[float, Dict[str, np.ndarray]] = {}

        for freq_hz in FREQUENCIES_HZ:
            frequency_samples[freq_hz] = {}
            for name, source in self.sources.items():
                s = source.get_samples(freq_hz, max_samples)
                if s is not None:
                    frequency_samples[freq_hz][name] = s

        # Combine all broadcasts
        return self.combiner.combine_all(frequency_samples, self.reference_source)

    def clear_samples(self) -> None:
        """Clear all captured samples."""
        for source in self.sources.values():
            source.clear_samples()

    @property
    def is_running(self) -> bool:
        """Whether the engine is running."""
        return self._running

    def get_status(self) -> Dict:
        """Get engine status."""
        return {
            "running": self._running,
            "sources": {
                name: {
                    "connected": source.is_connected,
                    "capturing": source.is_capturing,
                    "frequencies": source.frequencies,
                }
                for name, source in self.sources.items()
            },
            "reference_source": self.reference_source,
        }
