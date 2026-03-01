"""
PhaseEngine - Main orchestrator for coherent phased array processing.

Coordinates:
1. Multiple RadiodSources (one per physical radiod)
2. Sample alignment and calibration
3. Phase combining for all broadcasts
4. Virtual radio interface for downstream apps (hf-timestd)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import logging
import threading
import time

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


@dataclass(frozen=True)
class CalibrationResult:
    """Result of array calibration."""

    reference_source: str
    source_delays: Dict[str, int]  # source_name -> delay in samples
    source_phases: Dict[str, float]  # source_name -> phase in degrees
    correlation_coefficients: Dict[str, float]  # source_name -> correlation
    calibration_frequency_hz: float
    timestamp: float


class PhaseEngine:
    """
    Main orchestrator for the phased array processing system.

    Lifecycle:
        1. Create PhaseEngine with QTH and source configs
        2. Call connect() to connect to all radiods
        3. Call calibrate() to perform sample alignment
        4. Call start() to begin continuous processing
        5. Combined streams are available via get_combined_samples()
        6. Call stop() and disconnect() when done
    """

    def __init__(
        self,
        qth_latitude: float,
        qth_longitude: float,
        sources: List[SourceConfig],
        reference_source: Optional[str] = None,
        sample_rate: int = 12000,
        calibration_frequency_hz: float = 900e3,
    ):
        """
        Initialize the PhaseEngine.

        Args:
            qth_latitude: Observer latitude (degrees)
            qth_longitude: Observer longitude (degrees)
            sources: List of source configurations
            reference_source: Name of reference source (default: first source)
            sample_rate: Sample rate for all channels
            calibration_frequency_hz: Frequency for calibration (strong AM station)
        """
        self.qth_latitude = qth_latitude
        self.qth_longitude = qth_longitude
        self.sample_rate = sample_rate
        self.calibration_frequency_hz = calibration_frequency_hz

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

        # Cross-correlation aligner
        from .dsp.alignment import CrossCorrelator

        self.aligner = CrossCorrelator(sample_rate)

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

    def calibrate(
        self,
        frequency_hz: Optional[float] = None,
        duration_sec: float = 3.0,
    ) -> CalibrationResult:
        """
        Perform sample alignment calibration.

        Tunes all sources to a strong AM station and measures
        sample delays and phase offsets relative to the reference.

        Args:
            frequency_hz: Calibration frequency (default: self.calibration_frequency_hz)
            duration_sec: Capture duration in seconds

        Returns:
            CalibrationResult with delays and phases
        """
        frequency_hz = frequency_hz or self.calibration_frequency_hz

        logger.info(f"Starting calibration at {frequency_hz/1e3:.0f} kHz...")

        # Create calibration channels on all sources
        for source in self.sources.values():
            source.create_channel(frequency_hz)

        # Start capture
        for source in self.sources.values():
            source.start_capture([frequency_hz])

        # Wait for samples
        time.sleep(duration_sec + 0.5)

        # Stop capture
        for source in self.sources.values():
            source.stop_capture()

        # Get samples from each source
        samples = {}
        for name, source in self.sources.items():
            s = source.get_samples(frequency_hz)
            if s is not None and len(s) > 0:
                samples[name] = s
                logger.info(f"  {name}: {len(s)} samples, power={np.mean(np.abs(s)**2):.2e}")
            else:
                logger.warning(f"  {name}: No samples!")

        if self.reference_source not in samples:
            raise RuntimeError(f"Reference source {self.reference_source} has no samples")

        # Calculate delays and phases relative to reference
        ref_samples = samples[self.reference_source]
        source_delays = {self.reference_source: 0}
        source_phases = {self.reference_source: 0.0}
        correlations = {self.reference_source: 1.0}

        for name, s in samples.items():
            if name == self.reference_source:
                continue

            result = self.aligner.calibrate_pair(
                reference=ref_samples,
                target=s,
                reference_name=self.reference_source,
                target_name=name,
                frequency_hz=frequency_hz,
            )

            source_delays[name] = result.delay_samples
            source_phases[name] = np.degrees(result.phase_offset_rad)
            correlations[name] = result.correlation_score

            logger.info(
                f"  {name}: delay={result.delay_samples}, "
                f"phase={np.degrees(result.phase_offset_rad):.1f}°, "
                f"corr={result.correlation_score:.4f}"
            )

        # Remove calibration channels
        for source in self.sources.values():
            source.remove_channel(frequency_hz)

        # Store calibration
        self.calibration = CalibrationResult(
            reference_source=self.reference_source,
            source_delays=source_delays,
            source_phases=source_phases,
            correlation_coefficients=correlations,
            calibration_frequency_hz=frequency_hz,
            timestamp=time.time(),
        )

        # Create phase combiner with calibration
        self._create_combiner()

        logger.info("Calibration complete")
        return self.calibration

    def _create_combiner(self) -> None:
        """Create the phase combiner with current calibration."""
        # Apply calibration to combiner
        if self.calibration:
            # We need to construct a weights array matching self.array.antenna_names
            weights = np.ones(self.array.n_elements, dtype=np.complex64)
            for i, name in enumerate(self.array.antenna_names):
                # The alignment gives us phase_offset_rad relative to reference.
                # If target is delayed/phase-shifted by +theta, we need to apply -theta to correct it.
                phase_rad = np.radians(self.calibration.source_phases.get(name, 0.0))
                # Alternatively we can construct complex weight directly
                weights[i] = np.exp(-1j * phase_rad)

            self.combiner.set_calibration(weights)

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

    def open_channel(self, frequency_hz: float) -> None:
        """
        Lazily open a physical channel at the requested frequency on all sources.

        Called by VirtualChannelManager when a client (e.g. hf-timestd via
        ka9q-python) sends a TLV CMD requesting a frequency.  Creates the
        channel on every connected RadiodSource and starts capture so that
        combined samples are available for the egress loop.

        Args:
            frequency_hz: Center frequency in Hz

        Raises:
            RuntimeError: If not yet calibrated, or no sources are connected
        """
        if self.combiner is None:
            raise RuntimeError("Cannot open channel: engine is not calibrated")

        with self._lock:
            for name, source in self.sources.items():
                if not source.is_connected:
                    logger.warning(f"open_channel: source {name} not connected, skipping")
                    continue
                try:
                    if frequency_hz in source.frequencies:
                        logger.debug(f"open_channel: {name} already has {frequency_hz/1e6:.3f} MHz")
                        continue
                    source.create_channel(frequency_hz, sample_rate=self.sample_rate)
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

        if self.combiner is None:
            raise RuntimeError("Must calibrate before starting")

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
        virtual_channel: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Get mathematically combined samples for a virtual channel.

        Args:
            virtual_channel: The virtual channel config dict
            max_samples: Maximum number of samples to return

        Returns:
            Tuple of (combined complex sample array, RTP timestamp), or None if not available
        """
        if self.combiner is None:
            return None

        freq_hz = virtual_channel.get("frequency_hz")
        if not freq_hz:
            return None

        # Collect and consume samples from all sources at this frequency.
        # consume_samples() clears the buffer atomically so the egress loop
        # does not reprocess the same data on the next iteration.
        samples = {}
        authoritative_timestamp = None
        
        for name, source in self.sources.items():
            result = source.consume_samples(freq_hz, max_samples)
            if result is not None:
                s, ts = result
                samples[name] = s
                # Use the reference source's timestamp as the authoritative time
                # for the combined stream.
                if name == self.reference_source:
                    authoritative_timestamp = ts

        if not samples or authoritative_timestamp is None:
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
            return combined, authoritative_timestamp

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
        if self.combiner is None:
            return {}

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

    @property
    def is_calibrated(self) -> bool:
        """Whether calibration has been performed."""
        return self.calibration is not None

    def get_status(self) -> Dict:
        """Get engine status."""
        return {
            "running": self._running,
            "calibrated": self.is_calibrated,
            "sources": {
                name: {
                    "connected": source.is_connected,
                    "capturing": source.is_capturing,
                    "frequencies": source.frequencies,
                }
                for name, source in self.sources.items()
            },
            "reference_source": self.reference_source,
            "calibration": (
                {
                    "delays": self.calibration.source_delays,
                    "phases": self.calibration.source_phases,
                    "correlations": self.calibration.correlation_coefficients,
                    "frequency_hz": self.calibration.calibration_frequency_hz,
                }
                if self.calibration
                else None
            ),
        }
