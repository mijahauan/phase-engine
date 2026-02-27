"""
Calibration Sources - Auto-discovery of AM broadcast stations for array calibration.

Uses FCC CDBS data to find optimal local AM stations for geometric calibration.
Ground-wave signals from known tower locations provide stable reference bearings.
"""

import logging
import math
import os
import json
import urllib.request
import zipfile
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# FCC CDBS AM engineering data URL
FCC_AM_DATA_URL = "https://www.fcc.gov/file/download/file/am_eng_data.zip"

# Cache location
DEFAULT_CACHE_DIR = Path("/var/cache/phase-engine")
CACHE_FILENAME = "am_stations.json"
CACHE_MAX_AGE_DAYS = 7

# Clear channel frequencies (Class A) - avoid at night due to skywave interference
# These 50 kW stations can be heard across the continent at night
CLEAR_CHANNEL_FREQUENCIES_KHZ = frozenset({
    540, 640, 650, 660, 670, 680, 700, 710, 720, 750, 760, 770, 780,
    800, 820, 830, 840, 850, 870, 880, 890, 900, 940, 990, 1000, 1010,
    1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120,
    1130, 1140, 1160, 1170, 1180, 1190, 1200, 1210, 1220, 1500, 1510,
    1520, 1530, 1540, 1550, 1560, 1570, 1580
})


@dataclass
class AMStation:
    """An AM broadcast station suitable for calibration."""
    call_sign: str
    frequency_khz: int
    latitude: float
    longitude: float
    power_kw: float
    city: str
    state: str
    hours_operation: str  # "U" = unlimited, "D" = daytime, "N" = nighttime
    is_directional: bool
    
    # Computed relative to QTH
    distance_km: float = 0.0
    azimuth_deg: float = 0.0
    
    def is_clear_channel(self) -> bool:
        """Check if this station is on a clear channel frequency."""
        return self.frequency_khz in CLEAR_CHANNEL_FREQUENCIES_KHZ
    
    def is_safe_for_night(self) -> bool:
        """Check if this station is safe for nighttime calibration."""
        # Avoid clear channels at night (skywave interference)
        # Prefer non-directional stations (predictable pattern)
        return not self.is_clear_channel() and not self.is_directional
    
    def is_available_now(self) -> bool:
        """Check if station is currently on-air based on hours_operation."""
        if self.hours_operation == "U":  # Unlimited
            return True
        
        # Get current local hour (simplified - assumes local time)
        hour = datetime.now().hour
        is_daytime = 6 <= hour < 18
        
        if self.hours_operation == "D":  # Daytime only
            return is_daytime
        elif self.hours_operation == "N":  # Nighttime only
            return not is_daytime
        
        return True  # Default to available


@dataclass
class CalibrationSourceSet:
    """A set of AM stations selected for array calibration."""
    stations: list[AMStation]
    qth_latitude: float
    qth_longitude: float
    selection_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def azimuth_spread(self) -> float:
        """Calculate the angular spread of the selected stations."""
        if len(self.stations) < 2:
            return 0.0
        
        azimuths = sorted(s.azimuth_deg for s in self.stations)
        
        # Find maximum gap (including wrap-around)
        max_gap = 0.0
        for i in range(len(azimuths)):
            next_i = (i + 1) % len(azimuths)
            gap = (azimuths[next_i] - azimuths[i]) % 360
            max_gap = max(max_gap, gap)
        
        # Spread is 360 minus the largest gap
        return 360.0 - max_gap
    
    def is_good_geometry(self, min_spread_deg: float = 60.0) -> bool:
        """Check if the station geometry is good for calibration."""
        return len(self.stations) >= 2 and self.azimuth_spread >= min_spread_deg


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    R = 6371.0  # Earth radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def calculate_azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing from point 1 to point 2 in degrees."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    
    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360


class AMStationDatabase:
    """
    Database of AM broadcast stations from FCC CDBS.
    
    Downloads and caches FCC data, provides queries for finding
    calibration sources near a given location.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the station database.
        
        Args:
            cache_dir: Directory for caching FCC data
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.stations: list[AMStation] = []
        self._loaded = False
    
    def _cache_path(self) -> Path:
        return self.cache_dir / CACHE_FILENAME
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data exists and is recent enough."""
        cache_path = self._cache_path()
        if not cache_path.exists():
            return False
        
        # Check age
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        age = datetime.now(timezone.utc) - mtime
        return age.days < CACHE_MAX_AGE_DAYS
    
    def _load_from_cache(self) -> bool:
        """Load stations from cache file."""
        cache_path = self._cache_path()
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            self.stations = [
                AMStation(**s) for s in data.get('stations', [])
            ]
            logger.info(f"Loaded {len(self.stations)} stations from cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def _save_to_cache(self) -> None:
        """Save stations to cache file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path()
        
        data = {
            'updated': datetime.now(timezone.utc).isoformat(),
            'stations': [
                {
                    'call_sign': s.call_sign,
                    'frequency_khz': s.frequency_khz,
                    'latitude': s.latitude,
                    'longitude': s.longitude,
                    'power_kw': s.power_kw,
                    'city': s.city,
                    'state': s.state,
                    'hours_operation': s.hours_operation,
                    'is_directional': s.is_directional,
                }
                for s in self.stations
            ]
        }
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved {len(self.stations)} stations to cache")
    
    def _download_fcc_data(self) -> bool:
        """Download and parse FCC AM engineering data."""
        logger.info("Downloading FCC AM station data...")
        
        try:
            with urllib.request.urlopen(FCC_AM_DATA_URL, timeout=60) as response:
                zip_data = response.read()
            
            # Extract the pipe-delimited data file
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                # Find the .dat file
                dat_files = [n for n in zf.namelist() if n.endswith('.dat')]
                if not dat_files:
                    logger.error("No .dat file found in FCC zip")
                    return False
                
                with zf.open(dat_files[0]) as f:
                    content = f.read().decode('latin-1')
            
            self._parse_fcc_data(content)
            return True
            
        except Exception as e:
            logger.error(f"Failed to download FCC data: {e}")
            return False
    
    def _parse_fcc_data(self, content: str) -> None:
        """Parse FCC AM engineering data (pipe-delimited format)."""
        self.stations = []
        
        lines = content.strip().split('\n')
        if not lines:
            return
        
        # First line is header
        header = lines[0].split('|')
        
        # Find column indices (FCC format varies, so we search by name)
        col_map = {name.strip().lower(): i for i, name in enumerate(header)}
        
        # Required columns
        required = ['call_sign', 'frequency', 'lat_deg', 'lat_min', 'lat_sec', 
                    'lon_deg', 'lon_min', 'lon_sec', 'power', 'city', 'state']
        
        # Try alternate column names
        alt_names = {
            'call_sign': ['call_sign', 'callsign', 'call'],
            'frequency': ['frequency', 'freq'],
            'lat_deg': ['lat_deg', 'lat_d'],
            'lat_min': ['lat_min', 'lat_m'],
            'lat_sec': ['lat_sec', 'lat_s'],
            'lon_deg': ['lon_deg', 'long_deg', 'lon_d'],
            'lon_min': ['lon_min', 'long_min', 'lon_m'],
            'lon_sec': ['lon_sec', 'long_sec', 'lon_s'],
            'power': ['power', 'erp', 'ant_power'],
            'city': ['city', 'comm_city'],
            'state': ['state', 'comm_state'],
            'hours_operation': ['hours_operation', 'hours', 'oper_hours'],
            'dir_ind': ['dir_ind', 'directional', 'ant_dir_ind'],
        }
        
        def find_col(name: str) -> int:
            for alt in alt_names.get(name, [name]):
                if alt in col_map:
                    return col_map[alt]
            return -1
        
        indices = {name: find_col(name) for name in required + ['hours_operation', 'dir_ind']}
        
        # Check we have required columns
        missing = [n for n in required if indices[n] < 0]
        if missing:
            logger.warning(f"Missing columns in FCC data: {missing}")
            logger.info(f"Available columns: {list(col_map.keys())[:20]}...")
            return
        
        # Parse data rows
        for line in lines[1:]:
            fields = line.split('|')
            if len(fields) < max(indices.values()) + 1:
                continue
            
            try:
                call_sign = fields[indices['call_sign']].strip()
                frequency = int(fields[indices['frequency']].strip() or 0)
                
                lat_deg = float(fields[indices['lat_deg']].strip() or 0)
                lat_min = float(fields[indices['lat_min']].strip() or 0)
                lat_sec = float(fields[indices['lat_sec']].strip() or 0)
                
                lon_deg = float(fields[indices['lon_deg']].strip() or 0)
                lon_min = float(fields[indices['lon_min']].strip() or 0)
                lon_sec = float(fields[indices['lon_sec']].strip() or 0)
                
                power_str = fields[indices['power']].strip()
                power = float(power_str) if power_str else 0.0
                
                city = fields[indices['city']].strip()
                state = fields[indices['state']].strip()
                
                hours_idx = indices.get('hours_operation', -1)
                hours = fields[hours_idx].strip() if hours_idx >= 0 and hours_idx < len(fields) else 'U'
                
                dir_idx = indices.get('dir_ind', -1)
                dir_ind = fields[dir_idx].strip() if dir_idx >= 0 and dir_idx < len(fields) else ''
                is_directional = dir_ind.upper() in ('D', 'DA', 'DA1', 'DA2', 'DAN', 'DAD')
                
                # Convert to decimal degrees
                # US longitudes are West (negative)
                latitude = lat_deg + lat_min / 60 + lat_sec / 3600
                longitude = -(lon_deg + lon_min / 60 + lon_sec / 3600)
                
                # Skip invalid entries
                if not call_sign or frequency < 530 or frequency > 1700:
                    continue
                if latitude == 0 and longitude == 0:
                    continue
                
                # Power is in kW in FCC data
                power_kw = power
                
                station = AMStation(
                    call_sign=call_sign,
                    frequency_khz=frequency,
                    latitude=latitude,
                    longitude=longitude,
                    power_kw=power_kw,
                    city=city,
                    state=state,
                    hours_operation=hours or 'U',
                    is_directional=is_directional,
                )
                
                self.stations.append(station)
                
            except (ValueError, IndexError) as e:
                continue  # Skip malformed rows
        
        logger.info(f"Parsed {len(self.stations)} AM stations from FCC data")
    
    def load(self, force_refresh: bool = False) -> bool:
        """
        Load station database, from cache or FCC.
        
        Args:
            force_refresh: Force download even if cache is valid
            
        Returns:
            True if data was loaded successfully
        """
        if self._loaded and not force_refresh:
            return True
        
        # Try cache first
        if not force_refresh and self._is_cache_valid():
            if self._load_from_cache():
                self._loaded = True
                return True
        
        # Download from FCC
        if self._download_fcc_data():
            self._save_to_cache()
            self._loaded = True
            return True
        
        # Fall back to stale cache
        if self._load_from_cache():
            logger.warning("Using stale cache data")
            self._loaded = True
            return True
        
        return False
    
    def find_calibration_sources(
        self,
        qth_latitude: float,
        qth_longitude: float,
        max_distance_km: float = 50.0,
        min_power_kw: float = 5.0,
        count: int = 3,
        night_safe_only: bool = False,
        available_now: bool = True,
    ) -> CalibrationSourceSet:
        """
        Find optimal AM stations for array calibration.
        
        Selection criteria:
        1. Within max_distance_km (ground wave reliable)
        2. Above min_power_kw (strong signal)
        3. Maximize azimuth spread (good geometry)
        
        Args:
            qth_latitude: Receiver latitude
            qth_longitude: Receiver longitude
            max_distance_km: Maximum distance to consider
            min_power_kw: Minimum transmitter power
            count: Number of stations to select
            night_safe_only: Only select stations safe for nighttime
            available_now: Only select stations currently on-air
            
        Returns:
            CalibrationSourceSet with selected stations
        """
        if not self._loaded:
            self.load()
        
        # Calculate distance and azimuth for all stations
        candidates = []
        for station in self.stations:
            station.distance_km = haversine_distance(
                qth_latitude, qth_longitude,
                station.latitude, station.longitude
            )
            station.azimuth_deg = calculate_azimuth(
                qth_latitude, qth_longitude,
                station.latitude, station.longitude
            )
            
            # Apply filters
            if station.distance_km > max_distance_km:
                continue
            if station.power_kw < min_power_kw:
                continue
            if night_safe_only and not station.is_safe_for_night():
                continue
            if available_now and not station.is_available_now():
                continue
            
            candidates.append(station)
        
        logger.info(f"Found {len(candidates)} candidate stations within {max_distance_km} km")
        
        if len(candidates) <= count:
            return CalibrationSourceSet(
                stations=candidates,
                qth_latitude=qth_latitude,
                qth_longitude=qth_longitude,
            )
        
        # Select stations to maximize azimuth spread
        selected = self._select_for_spread(candidates, count)
        
        return CalibrationSourceSet(
            stations=selected,
            qth_latitude=qth_latitude,
            qth_longitude=qth_longitude,
        )
    
    def _select_for_spread(
        self,
        candidates: list[AMStation],
        count: int
    ) -> list[AMStation]:
        """Select stations to maximize azimuth spread."""
        if len(candidates) <= count:
            return candidates
        
        # Sort by power (prefer stronger stations)
        candidates = sorted(candidates, key=lambda s: -s.power_kw)
        
        # Greedy selection: pick stations that maximize angular separation
        selected = [candidates[0]]  # Start with strongest
        
        while len(selected) < count:
            best_candidate = None
            best_min_sep = -1
            
            for candidate in candidates:
                if candidate in selected:
                    continue
                
                # Calculate minimum separation from already-selected stations
                min_sep = min(
                    min(
                        abs(candidate.azimuth_deg - s.azimuth_deg),
                        360 - abs(candidate.azimuth_deg - s.azimuth_deg)
                    )
                    for s in selected
                )
                
                # Prefer larger minimum separation
                if min_sep > best_min_sep:
                    best_min_sep = min_sep
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
            else:
                break
        
        return selected


def find_calibration_sources(
    qth_latitude: float,
    qth_longitude: float,
    max_distance_km: float = 50.0,
    min_power_kw: float = 5.0,
    count: int = 3,
    cache_dir: Optional[Path] = None,
) -> CalibrationSourceSet:
    """
    Convenience function to find calibration sources.
    
    Args:
        qth_latitude: Receiver latitude
        qth_longitude: Receiver longitude
        max_distance_km: Maximum distance to consider
        min_power_kw: Minimum transmitter power
        count: Number of stations to select
        cache_dir: Cache directory for FCC data
        
    Returns:
        CalibrationSourceSet with selected stations
    """
    db = AMStationDatabase(cache_dir=cache_dir)
    db.load()
    
    # Check if it's nighttime
    hour = datetime.now().hour
    is_night = hour < 6 or hour >= 18
    
    return db.find_calibration_sources(
        qth_latitude=qth_latitude,
        qth_longitude=qth_longitude,
        max_distance_km=max_distance_km,
        min_power_kw=min_power_kw,
        count=count,
        night_safe_only=is_night,
        available_now=True,
    )
