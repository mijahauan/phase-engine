"""
Combiner - Coherent combination algorithms for phased array processing.

Implements various combination modes:
- MRC (Maximum Ratio Combining) - Maximize SNR
- EGC (Equal Gain Combining) - Simple sum
- Selection - Best antenna only
- Nulling - Cancel interference
- Adaptive - MVDR/LMS for automatic optimization
"""

import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)


class CombineMode(Enum):
    """Combination mode for the phased array."""
    MRC = "mrc"           # Maximum Ratio Combining (weight by SNR)
    EGC = "egc"           # Equal Gain Combining (simple average)
    SELECTION = "select"  # Use best antenna only
    NULLING = "null"      # Subtract to cancel interference
    ADAPTIVE = "adaptive" # MVDR/LMS automatic optimization
    MANUAL = "manual"     # User-specified phase weights


@dataclass
class CombinerState:
    """Current state of the combiner."""
    mode: CombineMode
    phase_weights_deg: Dict[str, float]  # Phase offset per antenna
    amplitude_weights: Dict[str, float]  # Amplitude weight per antenna
    reference_antenna: str
    null_depth_db: float = 0.0
    snr_improvement_db: float = 0.0


class CoherentCombiner:
    """
    Coherent signal combiner for phased array processing.
    
    Takes aligned I/Q streams from multiple antennas and produces
    a single combined output based on the selected mode.
    """
    
    def __init__(
        self,
        antenna_names: list[str],
        reference: str,
        sample_rate: int = 24000
    ):
        """
        Initialize the combiner.
        
        Args:
            antenna_names: Names of antennas in the array
            reference: Name of the reference antenna (phase = 0)
            sample_rate: Sample rate in Hz
        """
        self.antenna_names = antenna_names
        self.reference = reference
        self.sample_rate = sample_rate
        
        # Initialize weights (all equal, zero phase)
        self._phase_deg: Dict[str, float] = {name: 0.0 for name in antenna_names}
        self._amplitude: Dict[str, float] = {name: 1.0 for name in antenna_names}
        
        # Current mode
        self._mode = CombineMode.EGC
        
        # Adaptive algorithm state
        self._covariance_matrix: Optional[np.ndarray] = None
        self._adaptive_weights: Optional[np.ndarray] = None
        
        logger.info(f"CoherentCombiner: {len(antenna_names)} antennas, ref={reference}")
    
    @property
    def mode(self) -> CombineMode:
        return self._mode
    
    @mode.setter
    def mode(self, value: CombineMode) -> None:
        self._mode = value
        logger.info(f"Combiner mode set to: {value.value}")
    
    def set_phase(self, antenna: str, phase_deg: float) -> None:
        """Set phase offset for an antenna (degrees)."""
        if antenna in self._phase_deg:
            self._phase_deg[antenna] = phase_deg % 360
            logger.debug(f"Phase {antenna}: {phase_deg:.1f}°")
    
    def set_amplitude(self, antenna: str, amplitude: float) -> None:
        """Set amplitude weight for an antenna (0-1)."""
        if antenna in self._amplitude:
            self._amplitude[antenna] = max(0.0, min(1.0, amplitude))
    
    def rotate_phase(self, antenna: str, delta_deg: float) -> None:
        """Rotate phase by delta degrees."""
        if antenna in self._phase_deg:
            self._phase_deg[antenna] = (self._phase_deg[antenna] + delta_deg) % 360
    
    def combine(self, streams: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine multiple antenna streams into one.
        
        Args:
            streams: Dict mapping antenna names to complex sample arrays
            
        Returns:
            Combined complex sample array
        """
        if self._mode == CombineMode.EGC:
            return self._combine_egc(streams)
        elif self._mode == CombineMode.MRC:
            return self._combine_mrc(streams)
        elif self._mode == CombineMode.SELECTION:
            return self._combine_selection(streams)
        elif self._mode == CombineMode.NULLING:
            return self._combine_nulling(streams)
        elif self._mode == CombineMode.ADAPTIVE:
            return self._combine_adaptive(streams)
        elif self._mode == CombineMode.MANUAL:
            return self._combine_manual(streams)
        else:
            return self._combine_egc(streams)
    
    def _combine_egc(self, streams: Dict[str, np.ndarray]) -> np.ndarray:
        """Equal Gain Combining - simple average."""
        result = np.zeros_like(next(iter(streams.values())))
        count = 0
        
        for name, samples in streams.items():
            if name in self.antenna_names:
                result += samples
                count += 1
        
        if count > 0:
            result /= count
        
        return result
    
    def _combine_mrc(self, streams: Dict[str, np.ndarray]) -> np.ndarray:
        """Maximum Ratio Combining - weight by SNR."""
        # Estimate power (proxy for SNR) from each stream
        powers = {}
        for name, samples in streams.items():
            if name in self.antenna_names:
                powers[name] = np.mean(np.abs(samples) ** 2)
        
        total_power = sum(powers.values())
        if total_power < 1e-20:
            return self._combine_egc(streams)
        
        # Weight by normalized power
        result = np.zeros_like(next(iter(streams.values())))
        for name, samples in streams.items():
            if name in powers:
                weight = powers[name] / total_power
                # Apply phase correction relative to reference
                phase_rad = np.deg2rad(self._phase_deg.get(name, 0))
                result += weight * samples * np.exp(-1j * phase_rad)
        
        return result
    
    def _combine_selection(self, streams: Dict[str, np.ndarray]) -> np.ndarray:
        """Selection combining - use strongest antenna only."""
        best_name = None
        best_power = -1
        
        for name, samples in streams.items():
            if name in self.antenna_names:
                power = np.mean(np.abs(samples) ** 2)
                if power > best_power:
                    best_power = power
                    best_name = name
        
        if best_name:
            return streams[best_name].copy()
        else:
            return self._combine_egc(streams)
    
    def _combine_nulling(self, streams: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Nulling mode - subtract streams to cancel interference.
        
        Formula: s_ref - e^(jφ) * s_other
        """
        if self.reference not in streams:
            return self._combine_egc(streams)
        
        result = streams[self.reference].copy()
        
        for name, samples in streams.items():
            if name != self.reference and name in self.antenna_names:
                phase_rad = np.deg2rad(self._phase_deg.get(name, 0))
                result -= self._amplitude.get(name, 1.0) * samples * np.exp(1j * phase_rad)
        
        return result
    
    def _combine_manual(self, streams: Dict[str, np.ndarray]) -> np.ndarray:
        """Manual mode - use user-specified phase and amplitude weights."""
        result = np.zeros_like(next(iter(streams.values())))
        
        for name, samples in streams.items():
            if name in self.antenna_names:
                phase_rad = np.deg2rad(self._phase_deg.get(name, 0))
                amp = self._amplitude.get(name, 1.0)
                result += amp * samples * np.exp(-1j * phase_rad)
        
        # Normalize
        n_antennas = len([n for n in streams if n in self.antenna_names])
        if n_antennas > 0:
            result /= n_antennas
        
        return result
    
    def _combine_adaptive(self, streams: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Adaptive combining using MVDR (Minimum Variance Distortionless Response).
        
        Minimizes total output power while maintaining unity gain toward
        the signal of interest (assumed to be from reference direction).
        """
        # Stack streams into matrix: (n_antennas, n_samples)
        names = [n for n in self.antenna_names if n in streams]
        if len(names) < 2:
            return self._combine_egc(streams)
        
        X = np.vstack([streams[name] for name in names])
        n_antennas, n_samples = X.shape
        
        # Estimate covariance matrix with diagonal loading for stability
        R = (X @ X.conj().T) / n_samples
        diag_load = 0.01 * np.trace(R) / n_antennas
        R += diag_load * np.eye(n_antennas)
        
        # Steering vector (assume signal from reference direction = [1, 1, ..., 1])
        # In practice, this would be calculated from known geometry
        a = np.ones(n_antennas, dtype=np.complex64)
        
        # MVDR weights: w = R^(-1) * a / (a^H * R^(-1) * a)
        try:
            R_inv = np.linalg.inv(R)
            w = R_inv @ a / (a.conj().T @ R_inv @ a)
        except np.linalg.LinAlgError:
            logger.warning("MVDR matrix inversion failed, falling back to EGC")
            return self._combine_egc(streams)
        
        # Apply weights
        result = w.conj().T @ X
        
        # Store weights for diagnostics
        self._adaptive_weights = w
        
        return result.flatten()
    
    def null_now(self, streams: Dict[str, np.ndarray]) -> float:
        """
        Calculate optimal phase to null current interference.
        
        Assumes current signal is "bad" and finds phase that minimizes power.
        
        Args:
            streams: Current antenna streams
            
        Returns:
            Optimal null phase in degrees
        """
        if len(streams) < 2:
            return 0.0
        
        ref_stream = streams.get(self.reference)
        if ref_stream is None:
            return 0.0
        
        # Find the other antenna
        other_name = None
        other_stream = None
        for name, samples in streams.items():
            if name != self.reference and name in self.antenna_names:
                other_name = name
                other_stream = samples
                break
        
        if other_stream is None:
            return 0.0
        
        # Cross-correlation to find optimal phase
        cross = np.sum(ref_stream * other_stream.conj())
        optimal_phase_rad = np.angle(cross)
        optimal_phase_deg = np.rad2deg(optimal_phase_rad)
        
        # Set the phase for nulling (add 180° to invert)
        null_phase = (optimal_phase_deg + 180) % 360
        self.set_phase(other_name, null_phase)
        
        logger.info(f"Null phase set: {other_name} = {null_phase:.1f}°")
        return null_phase
    
    def get_state(self) -> CombinerState:
        """Get current combiner state."""
        return CombinerState(
            mode=self._mode,
            phase_weights_deg=dict(self._phase_deg),
            amplitude_weights=dict(self._amplitude),
            reference_antenna=self.reference,
        )
    
    def calculate_steering_phase(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        frequency_hz: float,
        positions_m: Dict[str, tuple[float, float, float]]
    ) -> Dict[str, float]:
        """
        Calculate phase weights to steer beam toward given direction.
        
        Args:
            azimuth_deg: Azimuth angle (0=North, 90=East)
            elevation_deg: Elevation angle (0=horizon, 90=zenith)
            frequency_hz: Operating frequency
            positions_m: Antenna positions {name: (x, y, z)}
            
        Returns:
            Dict of phase offsets in degrees
        """
        c = 299792458.0  # Speed of light
        wavelength = c / frequency_hz
        
        # Convert angles to radians
        az_rad = np.deg2rad(azimuth_deg)
        el_rad = np.deg2rad(elevation_deg)
        
        # Unit vector toward source
        ux = np.cos(el_rad) * np.sin(az_rad)
        uy = np.cos(el_rad) * np.cos(az_rad)
        uz = np.sin(el_rad)
        
        # Calculate phase for each antenna relative to reference
        ref_pos = positions_m.get(self.reference, (0, 0, 0))
        phases = {}
        
        for name, pos in positions_m.items():
            if name in self.antenna_names:
                # Path length difference
                dx = pos[0] - ref_pos[0]
                dy = pos[1] - ref_pos[1]
                dz = pos[2] - ref_pos[2]
                
                path_diff = dx * ux + dy * uy + dz * uz
                phase_rad = 2 * np.pi * path_diff / wavelength
                phases[name] = np.rad2deg(phase_rad) % 360
        
        return phases
