"""
Phase Engine - Coherent HF Phased Array Middleware

Transform multiple GPSDO-locked RX888 SDRs into a single, coherent
virtual receiver for diversity reception, beamforming, and interferometry.
"""

__version__ = "1.2.1"

from .engine import PhaseEngine, SourceConfig, CalibrationResult
from .client import (
    PhaseEngineControl,
    PhaseEngineStream,
    ChannelInfo,
    Capabilities,
    StreamQuality,
)

__all__ = [
    # Engine
    "PhaseEngine",
    "SourceConfig",
    "CalibrationResult",
    # Client API
    "PhaseEngineControl",
    "PhaseEngineStream",
    "ChannelInfo",
    "Capabilities",
    "StreamQuality",
]
