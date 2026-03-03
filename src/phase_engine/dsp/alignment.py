import numpy as np
from scipy import signal
from typing import Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentResult:
    delay_samples: int
    phase_offset_rad: float
    snr_db: float
    correlation_score: float


class CrossCorrelator:
    """
    Performs cross-correlation to find sample delay and phase offset
    between two streams.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def calibrate_pair(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        reference_name: str,
        target_name: str,
        frequency_hz: Optional[float] = None,
    ) -> AlignmentResult:
        """
        Cross-correlate target with reference to find delay and phase offset.
        """
        if len(reference) == 0 or len(target) == 0:
            raise ValueError("Empty signal provided for calibration")

        n = min(len(reference), len(target))
        ref = reference[:n]
        tgt = target[:n]

        # Calculate correlation
        # Use FFT for O(N log N) correlation instead of O(N^2)
        correlation = signal.correlate(tgt, ref, mode="full", method="fft")

        # Find peak
        peak_idx = np.argmax(np.abs(correlation))

        # The center index corresponds to 0 delay
        center_idx = len(ref) - 1

        # Delay is positive if target is delayed relative to reference
        delay_samples = peak_idx - center_idx

        # Phase offset is the angle of the correlation peak
        peak_val = correlation[peak_idx]
        phase_offset = np.angle(peak_val)

        score = np.abs(peak_val) / (np.linalg.norm(ref) * np.linalg.norm(tgt))

        # Calculate a rough SNR estimate
        snr_db = 10 * np.log10(score / (1 - score + 1e-10)) if score < 1 else 100.0

        logger.debug(
            f"Alignment {target_name} to {reference_name}: "
            f"delay={delay_samples} samples, phase={np.degrees(phase_offset):.1f}°, "
            f"score={score:.3f}"
        )

        return AlignmentResult(
            delay_samples=delay_samples,
            phase_offset_rad=phase_offset,
            snr_db=snr_db,
            correlation_score=score,
        )
