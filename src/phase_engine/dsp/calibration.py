from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class CalibrationResult:
    """Result of array calibration."""
    reference_source: str
    source_delays: Dict[str, int]
    source_phases: Dict[str, float]
    correlation_coefficients: Dict[str, float]
    calibration_frequency_hz: float
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "reference_source": self.reference_source,
            "source_delays": self.source_delays,
            "source_phases": self.source_phases,
            "correlation_coefficients": self.correlation_coefficients,
            "calibration_frequency_hz": self.calibration_frequency_hz,
            "timestamp": self.timestamp,
        }
