"""
Sample Alignment - Calibration routine for resolving integer sample ambiguity.

When multiple RX888 SDRs share a GPSDO clock but have no sync pulse,
each ADC starts at a random sample offset. This module provides
FFT-based cross-correlation to determine:
  1. Integer delay (Δn samples)
  2. Phase offset (Δφ degrees)
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of sample alignment calibration."""
    reference_antenna: str
    target_antenna: str
    delay_samples: int
    phase_offset_deg: float
    correlation_peak: float
    confidence: float  # 0-1, based on peak sharpness
    frequency_hz: float
    duration_seconds: float


class SampleAligner:
    """
    Calibration routine for determining sample delay and phase offset
    between antenna pairs.
    """
    
    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the sample aligner.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
    
    def calibrate_pair(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        reference_name: str = "ref",
        target_name: str = "target",
        frequency_hz: float = 0.0,
        max_delay_samples: Optional[int] = None
    ) -> CalibrationResult:
        """
        Calibrate delay and phase between two antenna streams.
        
        Uses FFT-based cross-correlation to find the integer sample
        delay, then refines the phase estimate.
        
        Args:
            reference: Complex samples from reference antenna
            target: Complex samples from target antenna
            reference_name: Name of reference antenna
            target_name: Name of target antenna
            frequency_hz: Center frequency (for logging)
            max_delay_samples: Maximum expected delay (default: 1 second)
            
        Returns:
            CalibrationResult with delay and phase offset
        """
        if max_delay_samples is None:
            max_delay_samples = self.sample_rate  # 1 second max
        
        n_samples = min(len(reference), len(target))
        duration = n_samples / self.sample_rate
        
        logger.info(f"Calibrating {target_name} relative to {reference_name}")
        logger.info(f"  Samples: {n_samples}, Duration: {duration:.2f}s")
        
        # Step 1: FFT-based cross-correlation for integer delay
        delay_samples, correlation_peak = self._find_integer_delay(
            reference[:n_samples],
            target[:n_samples],
            max_delay_samples
        )
        
        logger.info(f"  Integer delay: {delay_samples} samples "
                   f"({delay_samples / self.sample_rate * 1e6:.1f} µs)")
        
        # Step 2: Align streams and find phase offset
        if delay_samples >= 0:
            ref_aligned = reference[delay_samples:n_samples]
            tgt_aligned = target[:n_samples - delay_samples]
        else:
            ref_aligned = reference[:n_samples + delay_samples]
            tgt_aligned = target[-delay_samples:n_samples]
        
        phase_offset_deg = self._find_phase_offset(ref_aligned, tgt_aligned)
        
        logger.info(f"  Phase offset: {phase_offset_deg:.2f}°")
        
        # Step 3: Estimate confidence from correlation peak sharpness
        confidence = self._estimate_confidence(
            reference[:n_samples],
            target[:n_samples],
            delay_samples,
            max_delay_samples
        )
        
        logger.info(f"  Confidence: {confidence:.2f}")
        
        return CalibrationResult(
            reference_antenna=reference_name,
            target_antenna=target_name,
            delay_samples=delay_samples,
            phase_offset_deg=phase_offset_deg,
            correlation_peak=correlation_peak,
            confidence=confidence,
            frequency_hz=frequency_hz,
            duration_seconds=duration
        )
    
    def _find_integer_delay(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        max_delay: int
    ) -> Tuple[int, float]:
        """
        Find integer sample delay using FFT cross-correlation.
        
        Args:
            reference: Reference signal
            target: Target signal
            max_delay: Maximum delay to search
            
        Returns:
            (delay_samples, correlation_peak)
        """
        # Use scipy's correlate with FFT method for efficiency
        correlation = signal.correlate(target, reference, mode='full', method='fft')
        
        # The zero-lag position is at len(reference) - 1
        zero_lag = len(reference) - 1
        
        # Search within max_delay range
        search_start = max(0, zero_lag - max_delay)
        search_end = min(len(correlation), zero_lag + max_delay + 1)
        
        search_region = correlation[search_start:search_end]
        
        # Find peak (use magnitude for complex signals)
        peak_idx = np.argmax(np.abs(search_region))
        peak_value = np.abs(search_region[peak_idx])
        
        # Convert to delay (positive = target is delayed relative to reference)
        delay = (search_start + peak_idx) - zero_lag
        
        # Normalize peak value
        norm_factor = np.sqrt(np.sum(np.abs(reference)**2) * np.sum(np.abs(target)**2))
        if norm_factor > 0:
            peak_value /= norm_factor
        
        return int(delay), float(peak_value)
    
    def _find_phase_offset(
        self,
        reference: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        Find phase offset between aligned signals.
        
        Args:
            reference: Aligned reference signal
            target: Aligned target signal
            
        Returns:
            Phase offset in degrees
        """
        # Cross-correlation at zero lag gives phase relationship
        cross = np.sum(reference * target.conj())
        phase_rad = np.angle(cross)
        return np.rad2deg(phase_rad)
    
    def _estimate_confidence(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        found_delay: int,
        max_delay: int
    ) -> float:
        """
        Estimate confidence in the calibration result.
        
        Based on:
        - Sharpness of correlation peak
        - Signal coherence
        
        Returns:
            Confidence value 0-1
        """
        # Compute correlation around the found peak
        correlation = signal.correlate(target, reference, mode='full', method='fft')
        zero_lag = len(reference) - 1
        peak_idx = zero_lag + found_delay
        
        # Get peak value
        peak_value = np.abs(correlation[peak_idx])
        
        # Get average of nearby values (excluding peak)
        window = 10
        start = max(0, peak_idx - window)
        end = min(len(correlation), peak_idx + window + 1)
        nearby = np.abs(correlation[start:end])
        
        # Remove peak from average
        nearby_mean = (np.sum(nearby) - peak_value) / (len(nearby) - 1)
        
        # Peak-to-sidelobe ratio
        if nearby_mean > 0:
            psr = peak_value / nearby_mean
            # Map PSR to confidence (PSR of 10 = high confidence)
            confidence = min(1.0, (psr - 1) / 9)
        else:
            confidence = 1.0
        
        return max(0.0, confidence)
    
    def calibrate_array(
        self,
        streams: Dict[str, np.ndarray],
        reference_name: str,
        frequency_hz: float = 0.0
    ) -> Dict[str, CalibrationResult]:
        """
        Calibrate all antennas relative to a reference.
        
        Args:
            streams: Dict mapping antenna names to sample arrays
            reference_name: Name of reference antenna
            frequency_hz: Center frequency
            
        Returns:
            Dict mapping antenna names to CalibrationResult
        """
        if reference_name not in streams:
            raise ValueError(f"Reference antenna '{reference_name}' not in streams")
        
        reference = streams[reference_name]
        results = {}
        
        for name, samples in streams.items():
            if name == reference_name:
                # Reference has zero delay/phase by definition
                results[name] = CalibrationResult(
                    reference_antenna=reference_name,
                    target_antenna=name,
                    delay_samples=0,
                    phase_offset_deg=0.0,
                    correlation_peak=1.0,
                    confidence=1.0,
                    frequency_hz=frequency_hz,
                    duration_seconds=len(samples) / self.sample_rate
                )
            else:
                results[name] = self.calibrate_pair(
                    reference=reference,
                    target=samples,
                    reference_name=reference_name,
                    target_name=name,
                    frequency_hz=frequency_hz
                )
        
        return results


def run_calibration_routine(
    streams: Dict[str, np.ndarray],
    reference: str,
    sample_rate: int = 24000,
    frequency_hz: float = 0.0
) -> Dict[str, CalibrationResult]:
    """
    Convenience function to run full array calibration.
    
    Args:
        streams: Dict of antenna name -> complex samples
        reference: Name of reference antenna
        sample_rate: Sample rate in Hz
        frequency_hz: Center frequency for logging
        
    Returns:
        Dict of calibration results
    """
    aligner = SampleAligner(sample_rate=sample_rate)
    return aligner.calibrate_array(streams, reference, frequency_hz)
