"""
Client module - API-compatible interfaces for phase-engine.

Provides drop-in replacements for ka9q-python's RadiodControl and RadiodStream
with optional extensions for spatial filtering.
"""

from .control import (
    PhaseEngineControl,
    ChannelInfo,
    Capabilities,
    Target,
)
from .stream import (
    PhaseEngineStream,
    StreamQuality,
    SampleCallback,
)

__all__ = [
    # Control
    "PhaseEngineControl",
    "ChannelInfo",
    "Capabilities",
    "Target",
    # Stream
    "PhaseEngineStream",
    "StreamQuality",
    "SampleCallback",
]
