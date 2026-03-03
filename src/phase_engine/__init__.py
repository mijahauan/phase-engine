"""
Phase Engine - Coherent HF Phased Array Middleware

Transform multiple GPSDO-locked RX888 SDRs into a single, coherent
virtual receiver for diversity reception, beamforming, and interferometry.
"""

import os
# Limit numpy/scipy threading to prevent CPU explosion in the dataplane
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

__version__ = "1.3.0"

from .engine import PhaseEngine, SourceConfig
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
    # Client API
    "PhaseEngineControl",
    "PhaseEngineStream",
    "ChannelInfo",
    "Capabilities",
    "StreamQuality",
]
