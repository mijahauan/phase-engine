"""
PhaseEngine - Main orchestrator for coherent phased array processing.

Coordinates:
1. Multiple RadiodSources (one per physical radiod)
2. Sample alignment and calibration
3. Phase combining for all broadcasts
4. Virtual radio interface for downstream apps (hf-timestd)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
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

    def to_dict(self) -> dict:
        def clean_val(v):
            if isinstance(v, (np.integer, int)):
                return int(v)
            if isinstance(v, (np.floating, float)):
                if math.isnan(v):
                    return None
                return float(v)
            return v

        return {
            "reference_source": self.reference_source,
            "source_delays": {k: clean_val(v) for k, v in self.source_delays.items()},
            "source_phases": {k: clean_val(v) for k, v in self.source_phases.items()},
            "correlation_coefficients": {k: clean_val(v) for k, v in self.correlation_coefficients.items()},
            "calibration_frequency_hz": clean_val(self.calibration_frequency_hz),
            "timestamp": clean_val(self.timestamp),
        }


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
        
        # Track smoothed calibration weights per channel (frequency -> dict[antenna -> complex])
        self._channel_weights: Dict[float, Dict[str, complex]] = {}


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
        Applies continuous, per-channel cross-correlation phase alignment with EMA smoothing.
        """
        if self.combiner is None:
            return None

        freq_hz = virtual_channel.get("frequency_hz")
        if not freq_hz:
            return None

        samples = {}
        authoritative_timestamp = None
        
        for name, source in self.sources.items():
            result = source.consume_samples(freq_hz, max_samples)
            if result is not None:
                s, ts = result
                samples[name] = s
                if name == self.reference_source:
                    authoritative_timestamp = ts

        if not samples or authoritative_timestamp is None:
            return None
            
        # Initialize weight tracking for this frequency if needed
        if freq_hz not in self._channel_weights:
            self._channel_weights[freq_hz] = {name: 1.0 + 0j for name in self.array.antenna_names}
            
        # Per-channel continuous alignment
        ref_samples = samples.get(self.reference_source)
        if ref_samples is not None and len(ref_samples) > 0:
            calibration_weights = np.ones(self.array.n_elements, dtype=np.complex64)
            
            # EMA smoothing factor (e.g. 0.05 means 95% old, 5% new)
            # This integrates out fast ionospheric scintillation on skywaves
            alpha = 0.05 
            
            # If the correlation score drops below this, we reject the update and freeze the weights
            # This prevents the array from spinning wildly when WWV fades into the noise floor
            CORRELATION_THRESHOLD = 0.15
            
            for i, name in enumerate(self.array.antenna_names):
                if name == self.reference_source:
                    calibration_weights[i] = 1.0 + 0j
                    continue
                    
                tgt_samples = samples.get(name)
                if tgt_samples is not None and len(tgt_samples) > 0:
                    try:
                        res = self.aligner.calibrate_pair(
                            ref_samples, 
                            tgt_samples, 
                            self.reference_source, 
                            name, 
                            freq_hz
                        )
                        
                        if res.correlation_score >= CORRELATION_THRESHOLD:
                            # We apply the inverse phase to align the target to the reference
                            raw_weight = np.exp(-1j * res.phase_offset_rad)
                            
                            # Apply Exponential Moving Average (EMA) to the complex weight
                            old_weight = self._channel_weights[freq_hz][name]
                            smoothed_weight = (1 - alpha) * old_weight + alpha * raw_weight
                            
                            # Normalize to maintain unity gain
                            smoothed_weight /= np.abs(smoothed_weight)
                            
                            self._channel_weights[freq_hz][name] = smoothed_weight
                            calibration_weights[i] = smoothed_weight
                        else:
                            # Signal faded, fallback to last known good weight
                            logger.debug(f"Continuous alignment {name} at {freq_hz} DEGRADED (score={res.correlation_score:.3f}). Freezing weights.")
                            calibration_weights[i] = self._channel_weights[freq_hz][name]
                            
                    except Exception as e:
                        logger.warning(f"Continuous alignment failed for {name} at {freq_hz}: {e}")
                        # Fallback to last known good weight
                        calibration_weights[i] = self._channel_weights[freq_hz][name]
                else:
                    calibration_weights[i] = self._channel_weights[freq_hz][name]
            
            # Apply these dynamically calculated, smoothed weights
            self.combiner.set_calibration(calibration_weights)

        method = virtual_channel.get("combining_method", "mrc")
        bearing_deg = virtual_channel.get("bearing_deg", 0.0)

        try:
            a_target = None
            if method in ("beamform", "mvdr"):
                steer_az = np.radians(bearing_deg)
                el_rad = np.radians(10.0)
                k_vector = self.array.get_wave_vector(steer_az, el_rad, freq_hz)
                a_target = self.array.steering_vector(k_vector)

            combined = self.combiner.process(samples, method=method, steering_vector=a_target)
            return combined, authoritative_timestamp

        except Exception as e:
            logger.error(f"Error combining samples: {e}")
            return None

        freq_hz = virtual_channel.get("frequency_hz")
        if not freq_hz:
            return None

        samples = {}
        authoritative_timestamp = None
        
        for name, source in self.sources.items():
            result = source.consume_samples(freq_hz, max_samples)
            if result is not None:
                s, ts = result
                samples[name] = s
                if name == self.reference_source:
                    authoritative_timestamp = ts

        if not samples or authoritative_timestamp is None:
            return None
            
        # Initialize weight tracking for this frequency if needed
        if freq_hz not in self._channel_weights:
            self._channel_weights[freq_hz] = {name: 1.0 + 0j for name in self.array.antenna_names}
            
        # Per-channel continuous alignment
        ref_samples = samples.get(self.reference_source)
        if ref_samples is not None and len(ref_samples) > 0:
            calibration_weights = np.ones(self.array.n_elements, dtype=np.complex64)
            
            # EMA smoothing factor (e.g. 0.05 means 95% old, 5% new)
            # This integrates out fast ionospheric scintillation on skywaves
            alpha = 0.05 
            
            for i, name in enumerate(self.array.antenna_names):
                if name == self.reference_source:
                    calibration_weights[i] = 1.0 + 0j
                    continue
                    
                tgt_samples = samples.get(name)
                if tgt_samples is not None and len(tgt_samples) > 0:
                    try:
                        res = self.aligner.calibrate_pair(
                            ref_samples, 
                            tgt_samples, 
                            self.reference_source, 
                            name, 
                            freq_hz
                        )
                        # We apply the inverse phase to align the target to the reference
                        raw_weight = np.exp(-1j * res.phase_offset_rad)
                        
                        # Apply Exponential Moving Average (EMA) to the complex weight
                        old_weight = self._channel_weights[freq_hz][name]
                        smoothed_weight = (1 - alpha) * old_weight + alpha * raw_weight
                        
                        # Normalize to maintain unity gain
                        smoothed_weight /= np.abs(smoothed_weight)
                        
                        self._channel_weights[freq_hz][name] = smoothed_weight
                        calibration_weights[i] = smoothed_weight
                        
                    except Exception as e:
                        logger.warning(f"Continuous alignment failed for {name} at {freq_hz}: {e}")
                        # Fallback to last known good weight
                        calibration_weights[i] = self._channel_weights[freq_hz][name]
                else:
                    calibration_weights[i] = self._channel_weights[freq_hz][name]
            
            # Apply these dynamically calculated, smoothed weights
            self.combiner.set_calibration(calibration_weights)

        method = virtual_channel.get("combining_method", "mrc")
        bearing_deg = virtual_channel.get("bearing_deg", 0.0)

        try:
            a_target = None
            if method in ("beamform", "mvdr"):
                steer_az = np.radians(bearing_deg)
                el_rad = np.radians(10.0)
                k_vector = self.array.get_wave_vector(steer_az, el_rad, freq_hz)
                a_target = self.array.steering_vector(k_vector)

            combined = self.combiner.process(samples, method=method, steering_vector=a_target)
            return combined, authoritative_timestamp

        except Exception as e:
            logger.error(f"Error combining samples: {e}")
            return None

        freq_hz = virtual_channel.get("frequency_hz")
        if not freq_hz:
            return None

        samples = {}
        authoritative_timestamp = None
        
        for name, source in self.sources.items():
            result = source.consume_samples(freq_hz, max_samples)
            if result is not None:
                s, ts = result
                samples[name] = s
                if name == self.reference_source:
                    authoritative_timestamp = ts

        if not samples or authoritative_timestamp is None:
            return None
            
        # Per-channel continuous alignment
        # Extract the exact length block of samples we are about to process
        ref_samples = samples.get(self.reference_source)
        if ref_samples is not None and len(ref_samples) > 0:
            calibration_weights = np.ones(self.array.n_elements, dtype=np.complex64)
            for i, name in enumerate(self.array.antenna_names):
                if name == self.reference_source:
                    continue
                tgt_samples = samples.get(name)
                if tgt_samples is not None and len(tgt_samples) > 0:
                    try:
                        res = self.aligner.calibrate_pair(
                            ref_samples, 
                            tgt_samples, 
                            self.reference_source, 
                            name, 
                            freq_hz
                        )
                        # We apply the inverse phase to align the target to the reference
                        calibration_weights[i] = np.exp(-1j * res.phase_offset_rad)
                    except Exception as e:
                        logger.warning(f"Continuous alignment failed for {name} at {freq_hz}: {e}")
            
            # Apply these dynamically calculated weights
            self.combiner.set_calibration(calibration_weights)

        method = virtual_channel.get("combining_method", "mrc")
        bearing_deg = virtual_channel.get("bearing_deg", 0.0)

        try:
            a_target = None
            if method in ("beamform", "mvdr"):
                steer_az = np.radians(bearing_deg)
                el_rad = np.radians(10.0)
                k_vector = self.array.get_wave_vector(steer_az, el_rad, freq_hz)
                a_target = self.array.steering_vector(k_vector)

            combined = self.combiner.process(samples, method=method, steering_vector=a_target)
            return combined, authoritative_timestamp

        except Exception as e:
            logger.error(f"Error combining samples: {e}")
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
