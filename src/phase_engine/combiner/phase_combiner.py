"""
PhaseCombiner - Coherently combines streams from multiple radiod sources.

For each broadcast, applies:
1. Sample delay correction (from calibration)
2. Phase correction (from calibration + beam steering)
3. Coherent summation across all sources
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import threading
import logging

from ..sources.broadcasts import Broadcast, get_station_azimuth

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceCalibration:
    """Calibration data for a single source relative to reference."""
    source_name: str
    delay_samples: int = 0
    phase_offset_deg: float = 0.0
    amplitude_scale: float = 1.0
    

@dataclass(frozen=True)
class BroadcastCombiner:
    """
    Combiner for a single broadcast.
    
    Handles stream cloning for shared frequencies - the same input samples
    can be used with different phase weights for different stations.
    """
    broadcast: Broadcast
    steering_azimuth_deg: float = 0.0
    
    # Per-source calibration (source_name -> calibration)
    source_calibrations: Dict[str, SourceCalibration] = field(default_factory=dict)
    
    # Per-source steering phases (source_name -> phase in degrees)
    steering_phases: Dict[str, float] = field(default_factory=dict)
    
    def set_calibration(
        self,
        source_name: str,
        delay_samples: int,
        phase_offset_deg: float,
        amplitude_scale: float = 1.0,
    ) -> None:
        """Set calibration for a source."""
        self.source_calibrations[source_name] = SourceCalibration(
            source_name=source_name,
            delay_samples=delay_samples,
            phase_offset_deg=phase_offset_deg,
            amplitude_scale=amplitude_scale,
        )
        
    def set_steering_phase(self, source_name: str, phase_deg: float) -> None:
        """Set steering phase for a source."""
        self.steering_phases[source_name] = phase_deg
        
    def combine(
        self,
        samples: Dict[str, np.ndarray],
        reference_source: str,
    ) -> np.ndarray:
        """
        Combine samples from multiple sources.
        
        Args:
            samples: Dict mapping source_name -> complex sample array
            reference_source: Name of the reference source (delay=0, phase=0)
            
        Returns:
            Combined complex sample array
        """
        if not samples:
            return np.array([], dtype=np.complex64)
            
        # Find minimum length after delay alignment
        min_len = float('inf')
        for source_name, s in samples.items():
            cal = self.source_calibrations.get(source_name)
            delay = cal.delay_samples if cal else 0
            effective_len = len(s) - abs(delay)
            min_len = min(min_len, effective_len)
            
        if min_len <= 0:
            return np.array([], dtype=np.complex64)
            
        min_len = int(min_len)
        
        # Align and phase-correct each source
        aligned = []
        for source_name, s in samples.items():
            cal = self.source_calibrations.get(source_name)
            delay = cal.delay_samples if cal else 0
            phase_cal = cal.phase_offset_deg if cal else 0.0
            amplitude = cal.amplitude_scale if cal else 1.0
            
            # Apply delay
            if delay >= 0:
                s_aligned = s[delay:delay + min_len]
            else:
                s_aligned = s[:min_len]
                
            # Get steering phase
            phase_steer = self.steering_phases.get(source_name, 0.0)
            
            # Total phase correction: calibration + steering
            # Calibration phase aligns sources to reference
            # Steering phase points the beam
            total_phase = phase_cal + phase_steer
            
            # Apply phase and amplitude correction
            correction = amplitude * np.exp(1j * np.deg2rad(total_phase))
            s_corrected = s_aligned * correction
            
            aligned.append(s_corrected)
            
        # Coherent sum
        combined = np.sum(aligned, axis=0)
        
        return combined.astype(np.complex64)


class PhaseCombiner:
    """
    Manages phase combining for all broadcasts.
    
    Creates 17 BroadcastCombiners (one per broadcast) and handles
    stream cloning for shared frequencies.
    """
    
    def __init__(
        self,
        broadcasts: List[Broadcast],
        qth_latitude: float,
        qth_longitude: float,
        antenna_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ):
        """
        Initialize the phase combiner.
        
        Args:
            broadcasts: List of broadcasts to combine
            qth_latitude: Observer latitude (degrees)
            qth_longitude: Observer longitude (degrees)
            antenna_positions: Dict mapping source_name -> (x, y, z) in meters
        """
        self.broadcasts = broadcasts
        self.qth_latitude = qth_latitude
        self.qth_longitude = qth_longitude
        self.antenna_positions = antenna_positions or {}
        
        # Create a combiner for each broadcast
        self.combiners: Dict[Broadcast, BroadcastCombiner] = {}
        for broadcast in broadcasts:
            azimuth = get_station_azimuth(
                broadcast.station,
                qth_latitude,
                qth_longitude,
            )
            self.combiners[broadcast] = BroadcastCombiner(
                broadcast=broadcast,
                steering_azimuth_deg=azimuth,
            )
            
        self._lock = threading.Lock()
        
        logger.info(f"PhaseCombiner initialized with {len(broadcasts)} broadcasts")
        
    def set_source_calibration(
        self,
        source_name: str,
        delay_samples: int,
        phase_offset_deg: float,
        amplitude_scale: float = 1.0,
    ) -> None:
        """
        Set calibration for a source across all broadcasts.
        
        Args:
            source_name: Name of the source
            delay_samples: Sample delay relative to reference
            phase_offset_deg: Phase offset relative to reference
            amplitude_scale: Amplitude scaling factor
        """
        with self._lock:
            for combiner in self.combiners.values():
                combiner.set_calibration(
                    source_name,
                    delay_samples,
                    phase_offset_deg,
                    amplitude_scale,
                )
                
        logger.info(f"Set calibration for {source_name}: "
                   f"delay={delay_samples}, phase={phase_offset_deg:.1f}°")
                   
    def calculate_steering_phases(
        self,
        source_names: List[str],
        reference_source: str,
    ) -> None:
        """
        Calculate and set steering phases for all broadcasts.
        
        Uses antenna positions and broadcast azimuths to compute
        the phase weights needed to steer the beam toward each station.
        
        Args:
            source_names: List of source names
            reference_source: Name of the reference source
        """
        if not self.antenna_positions:
            logger.warning("No antenna positions set, using zero steering phases")
            return
            
        c = 299792458.0  # speed of light
        
        with self._lock:
            for broadcast, combiner in self.combiners.items():
                azimuth_rad = np.deg2rad(combiner.steering_azimuth_deg)
                wavelength = c / broadcast.frequency_hz
                
                # Unit vector toward station (in x-y plane, x=East, y=North)
                ux = np.sin(azimuth_rad)
                uy = np.cos(azimuth_rad)
                
                # Reference antenna position
                ref_pos = self.antenna_positions.get(reference_source, (0, 0, 0))
                
                for source_name in source_names:
                    if source_name == reference_source:
                        combiner.set_steering_phase(source_name, 0.0)
                        continue
                        
                    pos = self.antenna_positions.get(source_name, (0, 0, 0))
                    
                    # Relative position
                    dx = pos[0] - ref_pos[0]
                    dy = pos[1] - ref_pos[1]
                    
                    # Path length difference
                    path_diff = dx * ux + dy * uy
                    
                    # Phase difference
                    phase_rad = 2 * np.pi * path_diff / wavelength
                    phase_deg = np.rad2deg(phase_rad)
                    
                    combiner.set_steering_phase(source_name, -phase_deg)
                    
        logger.info(f"Calculated steering phases for {len(source_names)} sources")
        
    def combine_broadcast(
        self,
        broadcast: Broadcast,
        samples: Dict[str, np.ndarray],
        reference_source: str,
    ) -> np.ndarray:
        """
        Combine samples for a single broadcast.
        
        Args:
            broadcast: The broadcast to combine
            samples: Dict mapping source_name -> samples at broadcast frequency
            reference_source: Name of the reference source
            
        Returns:
            Combined sample array
        """
        combiner = self.combiners.get(broadcast)
        if combiner is None:
            raise ValueError(f"Unknown broadcast: {broadcast}")
            
        return combiner.combine(samples, reference_source)
        
    def combine_all(
        self,
        frequency_samples: Dict[float, Dict[str, np.ndarray]],
        reference_source: str,
    ) -> Dict[Broadcast, np.ndarray]:
        """
        Combine samples for all broadcasts.
        
        Args:
            frequency_samples: Dict mapping frequency_hz -> (source_name -> samples)
            reference_source: Name of the reference source
            
        Returns:
            Dict mapping Broadcast -> combined samples
        """
        results = {}
        
        for broadcast in self.broadcasts:
            freq_hz = broadcast.frequency_hz
            samples = frequency_samples.get(freq_hz, {})
            
            if not samples:
                logger.warning(f"No samples for {broadcast}")
                continue
                
            combined = self.combine_broadcast(broadcast, samples, reference_source)
            results[broadcast] = combined
            
        return results
        
    def get_broadcast_info(self, broadcast: Broadcast) -> Dict:
        """Get info about a broadcast combiner."""
        combiner = self.combiners.get(broadcast)
        if combiner is None:
            return {}
            
        return {
            "broadcast": str(broadcast),
            "azimuth_deg": combiner.steering_azimuth_deg,
            "calibrations": {
                name: {
                    "delay": cal.delay_samples,
                    "phase": cal.phase_offset_deg,
                    "amplitude": cal.amplitude_scale,
                }
                for name, cal in combiner.source_calibrations.items()
            },
            "steering_phases": dict(combiner.steering_phases),
        }
