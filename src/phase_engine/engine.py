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
import numpy as np
import threading
import time
import logging

from .sources import (
    RadiodSource,
    Broadcast,
    BROADCASTS,
    FREQUENCIES_HZ,
    get_broadcasts_for_frequency,
)
from .config_loader import get_engine_kwargs
from .dsp.array_geometry import AntennaArray
from .dsp.combiner import PhaseCombiner, SourceCalibration
from .calibration import SampleAligner

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """Configuration for a radiod source."""
    name: str
    status_address: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z in meters
    enabled: bool = True


@dataclass 
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
        
        logger.info(f"PhaseEngine initialized with {len(self.sources)} sources, "
                   f"reference={self.reference_source}")
        
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
            source_phases[name] = result.phase_offset_deg
            correlations[name] = result.correlation_peak
            
            logger.info(f"  {name}: delay={result.delay_samples}, "
                       f"phase={result.phase_offset_deg:.1f}°, "
                       f"corr={result.correlation_peak:.4f}")
                       
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
        self.combiner = PhaseCombiner(
            broadcasts=BROADCASTS,
            qth_latitude=self.qth_latitude,
            qth_longitude=self.qth_longitude,
            antenna_positions=self.source_positions,
        )
        
        # Apply calibration to combiner
        if self.calibration:
            for name in self.sources.keys():
                delay = self.calibration.source_delays.get(name, 0)
                phase = self.calibration.source_phases.get(name, 0.0)
                self.combiner.set_source_calibration(name, delay, phase)
                
        # Calculate steering phases
        self.combiner.calculate_steering_phases(
            list(self.sources.keys()),
            self.reference_source,
        )
        
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
        
    def start(self) -> None:
        """Start continuous capture on all broadcast frequencies."""
        if self._running:
            return
            
        if self.combiner is None:
            raise RuntimeError("Must calibrate before starting")
            
        # Create channels if not already done
        self.create_broadcast_channels()
        
        # Start capture on all sources
        for source in self.sources.values():
            source.start_capture(FREQUENCIES_HZ)
            
        self._running = True
        logger.info("PhaseEngine started")
        
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
    ) -> Optional[np.ndarray]:
        """
        Get combined samples for a virtual channel.
        
        Args:
            virtual_channel: Virtual channel state dictionary
            max_samples: Maximum samples to return
            
        Returns:
            Combined complex sample array, or None if not available
        """
        if self.combiner is None:
            return None
            
        broadcast = virtual_channel.get("broadcast")
        if not broadcast:
            return None
            
        freq_hz = broadcast.frequency_hz
        
        # Collect samples from all sources at this frequency
        samples = {}
        for name, source in self.sources.items():
            s = source.get_samples(freq_hz, max_samples)
            if s is not None:
                samples[name] = s
                
        if not samples:
            return None
            
        # Determine the combining method requested by the client
        method = virtual_channel.get("combining_method", "mrc")
        
        try:
            # Calculate true geographic bearing to the transmitter
            bearing_deg = 0.0
            if hasattr(broadcast, 'station') and broadcast.station:
                st_lat = broadcast.station.latitude
                st_lon = broadcast.station.longitude
                if st_lat != 0.0 or st_lon != 0.0:
                    bearing_deg = calculate_bearing(
                        self.qth_latitude, self.qth_longitude,
                        st_lat, st_lon
                    )
                    logger.debug(f"Calculated bearing to {broadcast.station.name}: {bearing_deg:.1f} deg")
            
            # If the method requires spatial steering (beamform or mvdr), compute the steering vector
            a_target = None
            if method in ["beamform", "mvdr", "null", "focus_null", "adaptive", "focus"]:
                if method in ["focus", "focus_null"]:
                    method = "mvdr"  # map phase-engine semantics to dsp implementations
                
                a_target = self.array.get_steering_vector(freq_hz, bearing_deg)
                
            combined = self.combiner.process(samples, method=method, steering_vector=a_target)
            return combined
            
        except Exception as e:
            logger.error(f"Combining failed for {broadcast.call_sign}: {e}")
            return None
            
        freq_hz = broadcast.frequency_hz
        
        # Collect samples from all sources at this frequency
        samples = {}
        for name, source in self.sources.items():
            s = source.get_samples(freq_hz, max_samples)
            if s is not None:
                samples[name] = s
                
        if not samples:
            return None
            
        # Determine the physical steering target from the virtual channel config
        # Currently, the proxy passes a generic broadcast request down.
        # We need to compute the true steering vector for this broadcast's station.
        
        # Calculate steering vector for target station
        target_station = broadcast.station
        
        try:
            # We need the true geographical azimuth to the station from the array center
            # For now, placeholder math: 0 degrees
            # TODO: Add Geographic math to calculate true bearing to target
            bearing_deg = 0.0
            
            # Use MVDR to aggressively null out other stations sharing this frequency
            method = "mvdr" 
            
            a_target = self.array.get_steering_vector(freq_hz, bearing_deg)
            combined = self.combiner.process(samples, method=method, steering_vector=a_target)
            return combined
            
        except Exception as e:
            logger.error(f"Combining failed for {broadcast.call_sign}: {e}")
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
            "calibration": {
                "delays": self.calibration.source_delays,
                "phases": self.calibration.source_phases,
                "correlations": self.calibration.correlation_coefficients,
                "frequency_hz": self.calibration.calibration_frequency_hz,
            } if self.calibration else None,
        }
