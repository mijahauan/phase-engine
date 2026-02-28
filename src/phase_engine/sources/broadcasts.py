"""
Broadcast configuration for time standard stations.

Defines the 17 broadcasts from 4 stations on 9 frequencies,
including station locations for beam steering.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


@dataclass(frozen=True)
class Station:
    """Time standard station definition."""

    call_sign: str
    name: str
    latitude: float  # degrees N
    longitude: float  # degrees E (negative for W)
    country: str


@dataclass(frozen=True)
class Broadcast:
    """A single broadcast (station + frequency)."""

    station: Station
    frequency_hz: float

    @property
    def call_sign(self) -> str:
        return self.station.call_sign

    @property
    def frequency_mhz(self) -> float:
        return self.frequency_hz / 1e6

    def __repr__(self) -> str:
        return f"Broadcast({self.station.call_sign}, {self.frequency_mhz:.3f} MHz)"


# Station definitions
WWV = Station(
    call_sign="WWV",
    name="WWV Fort Collins",
    latitude=40.6781,
    longitude=-105.0469,
    country="USA",
)

WWVH = Station(
    call_sign="WWVH",
    name="WWVH Kauai",
    latitude=21.9886,
    longitude=-159.7642,
    country="USA",
)

BPM = Station(
    call_sign="BPM",
    name="BPM Pucheng",
    latitude=34.9500,
    longitude=109.5500,
    country="China",
)

CHU = Station(
    call_sign="CHU",
    name="CHU Ottawa",
    latitude=45.2950,
    longitude=-75.7533,
    country="Canada",
)

# All stations
STATIONS: Dict[str, Station] = {
    "WWV": WWV,
    "WWVH": WWVH,
    "BPM": BPM,
    "CHU": CHU,
}

# Broadcast definitions (17 total)
BROADCASTS: List[Broadcast] = [
    # WWV broadcasts (5 frequencies)
    Broadcast(WWV, 2.5e6),
    Broadcast(WWV, 5.0e6),
    Broadcast(WWV, 10.0e6),
    Broadcast(WWV, 15.0e6),
    Broadcast(WWV, 20.0e6),
    # WWVH broadcasts (4 frequencies)
    Broadcast(WWVH, 2.5e6),
    Broadcast(WWVH, 5.0e6),
    Broadcast(WWVH, 10.0e6),
    Broadcast(WWVH, 15.0e6),
    # BPM broadcasts (5 frequencies)
    Broadcast(BPM, 2.5e6),
    Broadcast(BPM, 5.0e6),
    Broadcast(BPM, 10.0e6),
    Broadcast(BPM, 15.0e6),
    Broadcast(BPM, 16.2e6),
    # CHU broadcasts (3 frequencies)
    Broadcast(CHU, 3.33e6),
    Broadcast(CHU, 7.85e6),
    Broadcast(CHU, 14.67e6),
]

# Unique frequencies (9 total)
FREQUENCIES_HZ: List[float] = sorted(set(b.frequency_hz for b in BROADCASTS))

# Shared frequencies (4 frequencies with multiple stations)
SHARED_FREQUENCIES_HZ: List[float] = [2.5e6, 5.0e6, 10.0e6, 15.0e6]

# Unique frequencies (5 frequencies with single station)
UNIQUE_FREQUENCIES_HZ: List[float] = [3.33e6, 7.85e6, 14.67e6, 16.2e6, 20.0e6]


def get_broadcasts_for_frequency(frequency_hz: float) -> List[Broadcast]:
    """Get all broadcasts on a given frequency."""
    return [b for b in BROADCASTS if b.frequency_hz == frequency_hz]


def get_broadcasts_for_station(call_sign: str) -> List[Broadcast]:
    """Get all broadcasts from a given station."""
    return [b for b in BROADCASTS if b.station.call_sign == call_sign]


def calculate_azimuth(
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
) -> float:
    """
    Calculate azimuth (bearing) from one point to another.

    Args:
        from_lat: Latitude of origin (degrees)
        from_lon: Longitude of origin (degrees)
        to_lat: Latitude of destination (degrees)
        to_lon: Longitude of destination (degrees)

    Returns:
        Azimuth in degrees (0-360, 0=North, 90=East)
    """
    lat1 = math.radians(from_lat)
    lat2 = math.radians(to_lat)
    dlon = math.radians(to_lon - from_lon)

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    azimuth = math.degrees(math.atan2(x, y))
    return (azimuth + 360) % 360


def get_station_azimuth(
    station: Station,
    qth_latitude: float,
    qth_longitude: float,
) -> float:
    """
    Calculate azimuth from QTH to a station.

    Args:
        station: Target station
        qth_latitude: Observer latitude (degrees)
        qth_longitude: Observer longitude (degrees)

    Returns:
        Azimuth in degrees
    """
    return calculate_azimuth(
        qth_latitude,
        qth_longitude,
        station.latitude,
        station.longitude,
    )


def get_broadcast_azimuths(
    qth_latitude: float,
    qth_longitude: float,
) -> Dict[Broadcast, float]:
    """
    Calculate azimuths to all broadcasts from QTH.

    Args:
        qth_latitude: Observer latitude (degrees)
        qth_longitude: Observer longitude (degrees)

    Returns:
        Dict mapping Broadcast -> azimuth in degrees
    """
    return {
        broadcast: get_station_azimuth(broadcast.station, qth_latitude, qth_longitude)
        for broadcast in BROADCASTS
    }
