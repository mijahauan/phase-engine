"""
Terrestrial Calibration - Geometric array calibration using known AM broadcast sources.

Uses ground-wave signals from local AM stations with known tower locations
to calibrate the phased array. The known arrival angles allow solving for
inter-antenna delay and phase offsets.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from scipy import signal

from .sources import AMStation, CalibrationSourceSet, find_calibration_sources
from .sample_align import CalibrationResult

logger = logging.getLogger(__name__)


@dataclass
class TerrestrialCalibrationResult:
    """Result of terrestrial (geometric) calibration."""
    antenna_name: str
    delay_samples: int
    phase_offset_deg: float
    residual_error_deg: float
    sources_used: list[str]
    confidence: float


@dataclass 
class ArrayCalibrationResult:
    """Complete array calibration from terrestrial sources."""
    results: Dict[str, TerrestrialCalibrationResult]
    reference_antenna: str
    sources: CalibrationSourceSet
    rms_error_deg: float
    success: bool
    message: str


def predict_phase_from_geometry(
    azimuth_deg: float,
    elevation_deg: float,
    frequency_hz: float,
    antenna_positions: Dict[str, Tuple[float, float, float]],
    reference: str,
) -> Dict[str, float]:
    """
    Predict inter-antenna phase differences for a signal from a known direction.
    
    Args:
        azimuth_deg: Signal arrival azimuth (0=North, 90=East)
        elevation_deg: Signal arrival elevation (0=horizon, 90=zenith)
        frequency_hz: Signal frequency
        antenna_positions: Dict of antenna name -> (x, y, z) in meters
                          x=East, y=North, z=Up
        reference: Name of reference antenna
        
    Returns:
        Dict of antenna name -> predicted phase offset in degrees
    """
    c = 299792458.0  # Speed of light
    wavelength = c / frequency_hz
    
    # Convert angles to radians
    az_rad = np.deg2rad(azimuth_deg)
    el_rad = np.deg2rad(elevation_deg)
    
    # Unit vector FROM source direction (toward array)
    # Azimuth: 0=North, 90=East
    ux = np.cos(el_rad) * np.sin(az_rad)  # East component
    uy = np.cos(el_rad) * np.cos(az_rad)  # North component
    uz = np.sin(el_rad)                    # Up component
    
    # Reference position
    ref_pos = antenna_positions.get(reference, (0, 0, 0))
    
    phases = {}
    for name, pos in antenna_positions.items():
        # Position relative to reference
        dx = pos[0] - ref_pos[0]
        dy = pos[1] - ref_pos[1]
        dz = pos[2] - ref_pos[2]
        
        # Path length difference (positive = signal arrives later)
        path_diff = dx * ux + dy * uy + dz * uz
        
        # Phase difference
        phase_rad = 2 * np.pi * path_diff / wavelength
        phases[name] = np.rad2deg(phase_rad)
    
    return phases


def measure_phase_difference(
    reference_samples: np.ndarray,
    target_samples: np.ndarray,
) -> Tuple[float, float]:
    """
    Measure phase difference between two antenna streams.
    
    Args:
        reference_samples: Complex samples from reference antenna
        target_samples: Complex samples from target antenna
        
    Returns:
        (phase_difference_deg, coherence)
    """
    # Cross-correlation at zero lag
    cross = np.sum(reference_samples * target_samples.conj())
    
    # Phase is the angle of the cross-correlation
    phase_rad = np.angle(cross)
    phase_deg = np.rad2deg(phase_rad)
    
    # Coherence (0-1) indicates quality of measurement
    ref_power = np.sum(np.abs(reference_samples) ** 2)
    tgt_power = np.sum(np.abs(target_samples) ** 2)
    coherence = np.abs(cross) / np.sqrt(ref_power * tgt_power)
    
    return phase_deg, coherence


class TerrestrialCalibrator:
    """
    Calibrate phased array using known terrestrial AM broadcast sources.
    
    The calibration process:
    1. Tune to each calibration source frequency
    2. Measure inter-antenna phase differences
    3. Compare to predicted phases from geometry
    4. Solve for calibration constants (delay + phase offset per antenna)
    """
    
    def __init__(
        self,
        antenna_positions: Dict[str, Tuple[float, float, float]],
        reference_antenna: str,
        sample_rate: int = 24000,
    ):
        """
        Initialize the terrestrial calibrator.
        
        Args:
            antenna_positions: Dict of antenna name -> (x, y, z) in meters
            reference_antenna: Name of reference antenna (phase = 0)
            sample_rate: Sample rate in Hz
        """
        self.antenna_positions = antenna_positions
        self.reference = reference_antenna
        self.sample_rate = sample_rate
        
        self.antenna_names = list(antenna_positions.keys())
        if reference_antenna not in self.antenna_names:
            raise ValueError(f"Reference antenna '{reference_antenna}' not in positions")
    
    def calibrate_with_sources(
        self,
        sources: CalibrationSourceSet,
        streams_by_source: Dict[str, Dict[str, np.ndarray]],
    ) -> ArrayCalibrationResult:
        """
        Calibrate array using measured data from known sources.
        
        Args:
            sources: CalibrationSourceSet with station info
            streams_by_source: Dict of source_call_sign -> {antenna_name -> samples}
            
        Returns:
            ArrayCalibrationResult with calibration constants
        """
        if len(sources.stations) < 2:
            return ArrayCalibrationResult(
                results={},
                reference_antenna=self.reference,
                sources=sources,
                rms_error_deg=float('inf'),
                success=False,
                message="Need at least 2 calibration sources",
            )
        
        # Collect measurements and predictions for each source
        measurements = []  # List of (antenna, source, measured_phase)
        predictions = []   # List of (antenna, source, predicted_phase)
        
        for source in sources.stations:
            if source.call_sign not in streams_by_source:
                logger.warning(f"No data for source {source.call_sign}")
                continue
            
            streams = streams_by_source[source.call_sign]
            if self.reference not in streams:
                logger.warning(f"No reference antenna data for {source.call_sign}")
                continue
            
            ref_samples = streams[self.reference]
            
            # Predict phases from geometry
            # Ground wave: elevation ≈ 0°
            predicted = predict_phase_from_geometry(
                azimuth_deg=source.azimuth_deg,
                elevation_deg=0.0,
                frequency_hz=source.frequency_khz * 1000,
                antenna_positions=self.antenna_positions,
                reference=self.reference,
            )
            
            # Measure phases
            for antenna in self.antenna_names:
                if antenna == self.reference:
                    continue
                if antenna not in streams:
                    continue
                
                measured_phase, coherence = measure_phase_difference(
                    ref_samples, streams[antenna]
                )
                
                if coherence < 0.5:
                    logger.warning(f"Low coherence ({coherence:.2f}) for "
                                 f"{antenna} on {source.call_sign}")
                    continue
                
                measurements.append((antenna, source.call_sign, measured_phase))
                predictions.append((antenna, source.call_sign, predicted[antenna]))
        
        if len(measurements) < len(self.antenna_names) - 1:
            return ArrayCalibrationResult(
                results={},
                reference_antenna=self.reference,
                sources=sources,
                rms_error_deg=float('inf'),
                success=False,
                message=f"Insufficient measurements: {len(measurements)}",
            )
        
        # Solve for calibration offsets
        # For each antenna: measured = predicted + offset
        # offset = mean(measured - predicted) across all sources
        
        results = {}
        total_sq_error = 0.0
        n_measurements = 0
        
        for antenna in self.antenna_names:
            if antenna == self.reference:
                results[antenna] = TerrestrialCalibrationResult(
                    antenna_name=antenna,
                    delay_samples=0,
                    phase_offset_deg=0.0,
                    residual_error_deg=0.0,
                    sources_used=[],
                    confidence=1.0,
                )
                continue
            
            # Get all measurements for this antenna
            ant_measurements = [
                (src, meas, pred)
                for (ant, src, meas), (_, _, pred) in zip(measurements, predictions)
                if ant == antenna
            ]
            
            if not ant_measurements:
                logger.warning(f"No measurements for antenna {antenna}")
                continue
            
            # Calculate offset as mean of (measured - predicted)
            offsets = []
            for src, meas, pred in ant_measurements:
                # Handle phase wrapping
                diff = meas - pred
                while diff > 180:
                    diff -= 360
                while diff < -180:
                    diff += 360
                offsets.append(diff)
            
            phase_offset = np.mean(offsets)
            
            # Calculate residual error
            residuals = [off - phase_offset for off in offsets]
            residual_rms = np.sqrt(np.mean(np.array(residuals) ** 2))
            
            total_sq_error += sum(r ** 2 for r in residuals)
            n_measurements += len(residuals)
            
            # Estimate delay in samples (for sub-sample alignment)
            # At baseband, phase offset corresponds to time delay
            # This is approximate - true delay requires wideband measurement
            delay_samples = 0  # Integer delay from sample_align module
            
            # Confidence based on consistency across sources
            confidence = max(0.0, 1.0 - residual_rms / 30.0)  # 30° error = 0 confidence
            
            results[antenna] = TerrestrialCalibrationResult(
                antenna_name=antenna,
                delay_samples=delay_samples,
                phase_offset_deg=phase_offset,
                residual_error_deg=residual_rms,
                sources_used=[src for src, _, _ in ant_measurements],
                confidence=confidence,
            )
            
            logger.info(f"Calibration {antenna}: offset={phase_offset:.1f}°, "
                       f"residual={residual_rms:.1f}°, confidence={confidence:.2f}")
        
        rms_error = np.sqrt(total_sq_error / n_measurements) if n_measurements > 0 else float('inf')
        
        success = rms_error < 15.0 and len(results) == len(self.antenna_names)
        
        return ArrayCalibrationResult(
            results=results,
            reference_antenna=self.reference,
            sources=sources,
            rms_error_deg=rms_error,
            success=success,
            message="Calibration successful" if success else f"High residual error: {rms_error:.1f}°",
        )


def run_terrestrial_calibration(
    qth_latitude: float,
    qth_longitude: float,
    antenna_positions: Dict[str, Tuple[float, float, float]],
    reference_antenna: str,
    tune_and_capture_fn,
    sample_rate: int = 24000,
    capture_duration_s: float = 1.0,
    max_distance_km: float = 50.0,
    min_power_kw: float = 5.0,
) -> ArrayCalibrationResult:
    """
    Run complete terrestrial calibration routine (two-stage).
    
    Stage 1: Sample Alignment
        - Tune to strongest AM source
        - Cross-correlate to find integer sample delays between antennas
        - These delays are applied before phase measurement
    
    Stage 2: Phase Calibration  
        - Tune to each AM source
        - Measure inter-antenna phase differences
        - Compare to predicted phases from geometry
        - Solve for residual phase offsets
    
    Args:
        qth_latitude: Receiver latitude
        qth_longitude: Receiver longitude
        antenna_positions: Dict of antenna name -> (x, y, z) in meters
        reference_antenna: Name of reference antenna
        tune_and_capture_fn: Callback to tune radios and capture samples
                            Signature: (frequency_hz, duration_s) -> Dict[str, np.ndarray]
        sample_rate: Sample rate in Hz
        capture_duration_s: Duration to capture per source
        max_distance_km: Maximum distance for calibration sources
        min_power_kw: Minimum power for calibration sources
        
    Returns:
        ArrayCalibrationResult with calibration constants (delay + phase per antenna)
    """
    from .sample_align import SampleAligner
    
    # Find calibration sources
    sources = find_calibration_sources(
        qth_latitude=qth_latitude,
        qth_longitude=qth_longitude,
        max_distance_km=max_distance_km,
        min_power_kw=min_power_kw,
        count=3,
    )
    
    if not sources.stations:
        return ArrayCalibrationResult(
            results={},
            reference_antenna=reference_antenna,
            sources=sources,
            rms_error_deg=float('inf'),
            success=False,
            message="No calibration sources found",
        )
    
    if not sources.is_good_geometry():
        logger.warning(f"Poor source geometry: spread={sources.azimuth_spread:.0f}°")
    
    logger.info(f"Using {len(sources.stations)} calibration sources:")
    for s in sources.stations:
        logger.info(f"  {s.call_sign} {s.frequency_khz} kHz @ {s.azimuth_deg:.0f}° "
                   f"({s.distance_km:.0f} km, {s.power_kw:.0f} kW)")
    
    # =========================================================================
    # STAGE 1: Sample Alignment (Integer Delay)
    # =========================================================================
    # Use the strongest/closest source for sample alignment
    # This resolves the "ADC start time ambiguity" between radiod instances
    
    alignment_source = max(sources.stations, key=lambda s: s.power_kw / (s.distance_km + 1))
    logger.info(f"Stage 1: Sample alignment using {alignment_source.call_sign} "
               f"({alignment_source.frequency_khz} kHz)")
    
    try:
        alignment_streams = tune_and_capture_fn(
            alignment_source.frequency_khz * 1000,
            capture_duration_s * 2,  # Longer capture for better correlation
        )
    except Exception as e:
        return ArrayCalibrationResult(
            results={},
            reference_antenna=reference_antenna,
            sources=sources,
            rms_error_deg=float('inf'),
            success=False,
            message=f"Failed to capture alignment source: {e}",
        )
    
    # Run sample alignment
    aligner = SampleAligner(sample_rate=sample_rate)
    sample_delays = {}  # antenna -> integer delay in samples
    
    if reference_antenna not in alignment_streams:
        return ArrayCalibrationResult(
            results={},
            reference_antenna=reference_antenna,
            sources=sources,
            rms_error_deg=float('inf'),
            success=False,
            message=f"Reference antenna '{reference_antenna}' not in captured streams",
        )
    
    ref_samples = alignment_streams[reference_antenna]
    
    for antenna, samples in alignment_streams.items():
        if antenna == reference_antenna:
            sample_delays[antenna] = 0
            logger.info(f"  {antenna}: delay=0 samples (reference)")
            continue
        
        result = aligner.calibrate_pair(
            reference=ref_samples,
            target=samples,
            reference_name=reference_antenna,
            target_name=antenna,
            frequency_hz=alignment_source.frequency_khz * 1000,
        )
        
        sample_delays[antenna] = result.delay_samples
        logger.info(f"  {antenna}: delay={result.delay_samples} samples "
                   f"({result.delay_samples / sample_rate * 1e6:.1f} µs), "
                   f"confidence={result.confidence:.2f}")
    
    # =========================================================================
    # STAGE 2: Phase Calibration (using aligned streams)
    # =========================================================================
    logger.info("Stage 2: Phase calibration using terrestrial sources")
    
    # Capture data from each source, applying sample alignment
    streams_by_source = {}
    
    for source in sources.stations:
        logger.info(f"  Capturing {source.call_sign} at {source.frequency_khz} kHz...")
        
        try:
            raw_streams = tune_and_capture_fn(
                source.frequency_khz * 1000,
                capture_duration_s,
            )
            
            # Apply sample alignment (shift streams to compensate for delays)
            aligned_streams = {}
            for antenna, samples in raw_streams.items():
                delay = sample_delays.get(antenna, 0)
                if delay > 0:
                    # Target is delayed relative to reference, trim start
                    aligned_streams[antenna] = samples[delay:]
                elif delay < 0:
                    # Target is ahead of reference, trim end
                    aligned_streams[antenna] = samples[:delay]
                else:
                    aligned_streams[antenna] = samples
            
            # Ensure all streams are same length after alignment
            min_len = min(len(s) for s in aligned_streams.values())
            aligned_streams = {k: v[:min_len] for k, v in aligned_streams.items()}
            
            streams_by_source[source.call_sign] = aligned_streams
            
        except Exception as e:
            logger.error(f"Failed to capture {source.call_sign}: {e}")
    
    # Run phase calibration
    calibrator = TerrestrialCalibrator(
        antenna_positions=antenna_positions,
        reference_antenna=reference_antenna,
        sample_rate=sample_rate,
    )
    
    result = calibrator.calibrate_with_sources(sources, streams_by_source)
    
    # Merge sample delays into results
    for antenna, cal_result in result.results.items():
        cal_result.delay_samples = sample_delays.get(antenna, 0)
    
    return result
