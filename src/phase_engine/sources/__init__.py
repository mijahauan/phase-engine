"""
Sources module - Manages radiod instances as sample sources.
"""

from .radiod_source import RadiodSource, ChannelConfig, ChannelState
from .broadcasts import (
    Station,
    Broadcast,
    STATIONS,
    BROADCASTS,
    FREQUENCIES_HZ,
    SHARED_FREQUENCIES_HZ,
    UNIQUE_FREQUENCIES_HZ,
    WWV,
    WWVH,
    BPM,
    CHU,
    get_broadcasts_for_frequency,
    get_broadcasts_for_station,
    get_station_azimuth,
    get_broadcast_azimuths,
    calculate_azimuth,
)

__all__ = [
    # RadiodSource
    "RadiodSource",
    "ChannelConfig", 
    "ChannelState",
    # Broadcasts
    "Station",
    "Broadcast",
    "STATIONS",
    "BROADCASTS",
    "FREQUENCIES_HZ",
    "SHARED_FREQUENCIES_HZ",
    "UNIQUE_FREQUENCIES_HZ",
    "WWV",
    "WWVH",
    "BPM",
    "CHU",
    "get_broadcasts_for_frequency",
    "get_broadcasts_for_station",
    "get_station_azimuth",
    "get_broadcast_azimuths",
    "calculate_azimuth",
]
