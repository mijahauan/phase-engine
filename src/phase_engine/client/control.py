"""
PhaseEngineControl - API-compatible replacement for RadiodControl.

Provides the same interface as ka9q-python's RadiodControl, with optional
extensions for spatial filtering (beamforming, nulling).

Usage:
    # Drop-in replacement for RadiodControl
    from phase_engine.client import PhaseEngineControl
    
    control = PhaseEngineControl("phase-engine-status.local")
    
    # Standard channel creation (works like RadiodControl)
    channel = control.create_channel(frequency_hz=10e6, preset="iq")
    
    # Extended channel creation (phase-engine features)
    channel = control.create_channel(
        frequency_hz=10e6,
        preset="iq",
        reception_mode="focus",
        target="WWV",
        null_targets=["BPM"],
        combining_method="mrc",
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import threading
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChannelInfo:
    """
    Extended channel information returned by PhaseEngineControl.
    
    Compatible with ka9q-python's channel info, with additional
    phase-engine fields.
    """
    # Standard ka9q-python fields
    ssrc: int
    frequency: float  # Hz
    multicast_address: str
    port: int
    sample_rate: int
    preset: str
    encoding: int
    
    # Phase-engine extensions
    reception_mode: str = "omni"
    beam_azimuth_deg: Optional[float] = None
    beam_target: Optional[str] = None
    null_azimuths_deg: List[float] = field(default_factory=list)
    null_targets: List[str] = field(default_factory=list)
    combining_method: str = "mrc"
    estimated_gain_db: float = 0.0
    estimated_null_depth_db: float = 0.0
    
    @property
    def frequency_hz(self) -> float:
        """Alias for frequency (compatibility)."""
        return self.frequency


@dataclass
class Capabilities:
    """Phase-engine capabilities."""
    backend: str = "phase-engine"
    version: str = "1.0.0"
    n_antennas: int = 1
    dof: int = 0
    modes: List[str] = field(default_factory=lambda: ["omni"])
    max_simultaneous_beams: int = 1
    max_nulls: int = 0
    can_focus_and_null: bool = False
    can_aoa_estimate: bool = False
    max_resolvable_sources: int = 0
    aoa_algorithms: List[str] = field(default_factory=list)
    combining_methods: List[str] = field(default_factory=lambda: ["mrc", "egc", "selection"])
    array_center_lat: float = 0.0
    array_center_lon: float = 0.0
    calibration: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend,
            "version": self.version,
            "n_antennas": self.n_antennas,
            "dof": self.dof,
            "modes": self.modes,
            "max_simultaneous_beams": self.max_simultaneous_beams,
            "max_nulls": self.max_nulls,
            "can_focus_and_null": self.can_focus_and_null,
            "can_aoa_estimate": self.can_aoa_estimate,
            "max_resolvable_sources": self.max_resolvable_sources,
            "aoa_algorithms": self.aoa_algorithms,
            "combining_methods": self.combining_methods,
            "array_center_lat": self.array_center_lat,
            "array_center_lon": self.array_center_lon,
            "calibration": self.calibration,
        }


# Target can be station name, azimuth, or (lat, lon) tuple
Target = Union[str, float, Tuple[float, float]]


class PhaseEngineControl:
    """
    Control interface for phase-engine, API-compatible with RadiodControl.
    
    This class can be used as a drop-in replacement for ka9q-python's
    RadiodControl. Standard parameters work identically; extended
    parameters enable spatial filtering features.
    
    Args:
        status_address: Phase-engine status multicast address
            (e.g., "phase-engine-status.local")
    """
    
    def __init__(self, status_address: str):
        """
        Initialize connection to phase-engine.
        
        Args:
            status_address: Phase-engine status address
        """
        self.status_address = status_address
        self._channels: Dict[int, ChannelInfo] = {}
        self._ssrc_counter = 0
        self._lock = threading.Lock()
        self._closed = False
        
        # These will be set by the engine when it initializes this control
        self._engine = None  # Reference to PhaseEngine instance
        self._capabilities: Optional[Capabilities] = None
        
        logger.info(f"PhaseEngineControl initialized for {status_address}")
        
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Query phase-engine capabilities.
        
        Returns:
            Dictionary of capabilities (see API spec)
        """
        if self._capabilities:
            return self._capabilities.to_dict()
            
        # Default capabilities if not connected to engine
        return Capabilities().to_dict()
        
    def create_channel(
        self,
        frequency_hz: float,
        preset: str = "iq",
        sample_rate: int = 12000,
        encoding: Optional[int] = None,
        agc_enable: int = 0,
        gain: float = 30.0,
        destination: Optional[str] = None,
        ssrc: Optional[int] = None,
        # Phase-engine extensions
        reception_mode: str = "auto",
        target: Optional[Target] = None,
        null_targets: Optional[List[Target]] = None,
        combining_method: str = "mrc",
        priority: int = 0,
        **kwargs,  # Accept and ignore unknown parameters for compatibility
    ) -> ChannelInfo:
        """
        Create a channel with optional spatial filtering.
        
        Standard parameters (compatible with RadiodControl):
            frequency_hz: Center frequency in Hz
            preset: Channel preset ("iq", "am", "usb", etc.)
            sample_rate: Sample rate in Hz
            encoding: Sample encoding (default: F32)
            agc_enable: Enable AGC (0 or 1)
            gain: Gain in dB
            destination: RTP destination address
            ssrc: Specific SSRC to use (auto-allocated if None)
            
        Phase-engine extensions (ignored by plain radiod):
            reception_mode: "omni", "focus", "null", "focus_null", "adaptive", "auto"
            target: Beam target - station name, azimuth (deg), or (lat, lon)
            null_targets: List of interference sources to null
            combining_method: "mrc", "egc", or "selection"
            priority: Resource allocation priority
            
        Returns:
            ChannelInfo with channel details
        """
        if self._closed:
            raise RuntimeError("Control is closed")
            
        with self._lock:
            # Allocate SSRC if not provided
            if ssrc is None:
                ssrc = self._allocate_ssrc()
                
            # Resolve target to azimuth
            beam_azimuth = None
            beam_target_name = None
            if target is not None:
                beam_azimuth, beam_target_name = self._resolve_target(target)
                
            # Resolve null targets
            null_azimuths = []
            null_target_names = []
            if null_targets:
                for nt in null_targets:
                    az, name = self._resolve_target(nt)
                    if az is not None:
                        null_azimuths.append(az)
                        null_target_names.append(name or str(nt))
                        
            # Determine effective reception mode
            effective_mode = self._resolve_reception_mode(
                reception_mode, target, null_targets, frequency_hz
            )
            
            # Create channel info
            # In a full implementation, this would communicate with the engine
            # to set up the actual channel. For now, we create the info structure.
            channel = ChannelInfo(
                ssrc=ssrc,
                frequency=frequency_hz,
                multicast_address=self._allocate_multicast_address(ssrc),
                port=5004,
                sample_rate=sample_rate,
                preset=preset,
                encoding=encoding or 4,  # F32
                reception_mode=effective_mode,
                beam_azimuth_deg=beam_azimuth,
                beam_target=beam_target_name,
                null_azimuths_deg=null_azimuths,
                null_targets=null_target_names,
                combining_method=combining_method,
            )
            
            self._channels[ssrc] = channel
            
            # If we have an engine reference, create the actual channel
            if self._engine is not None:
                self._engine._create_channel_internal(channel)
                
            logger.info(f"Created channel: {frequency_hz/1e6:.3f} MHz, "
                       f"mode={effective_mode}, target={beam_target_name}")
                       
            return channel
            
    def ensure_channel(
        self,
        frequency_hz: float,
        frequency_tolerance: float = 1.0,
        timeout: float = 10.0,
        **kwargs,
    ) -> ChannelInfo:
        """
        Ensure a channel exists at the given frequency.
        
        If a channel already exists within tolerance, returns it.
        Otherwise creates a new channel.
        
        Args:
            frequency_hz: Desired frequency
            frequency_tolerance: Tolerance in Hz
            timeout: Timeout in seconds
            **kwargs: Passed to create_channel
            
        Returns:
            ChannelInfo for existing or new channel
        """
        # Check for existing channel
        for channel in self._channels.values():
            if abs(channel.frequency - frequency_hz) <= frequency_tolerance:
                return channel
                
        # Create new channel
        return self.create_channel(frequency_hz=frequency_hz, **kwargs)
        
    def remove_channel(self, ssrc: int) -> None:
        """
        Remove a channel.
        
        Args:
            ssrc: SSRC of channel to remove
        """
        with self._lock:
            channel = self._channels.pop(ssrc, None)
            if channel:
                logger.info(f"Removed channel SSRC={ssrc}")
                
                # If we have an engine reference, remove from engine
                if self._engine is not None:
                    self._engine._remove_channel_internal(ssrc)
                    
    def reconfigure_channel(
        self,
        ssrc: int,
        reception_mode: Optional[str] = None,
        target: Optional[Target] = None,
        null_targets: Optional[List[Target]] = None,
        combining_method: Optional[str] = None,
    ) -> ChannelInfo:
        """
        Reconfigure spatial filtering on an active channel.
        
        Args:
            ssrc: SSRC of channel to reconfigure
            reception_mode: New reception mode
            target: New beam target
            null_targets: New null targets
            combining_method: New combining method
            
        Returns:
            Updated ChannelInfo
        """
        with self._lock:
            channel = self._channels.get(ssrc)
            if channel is None:
                raise ValueError(f"Unknown channel SSRC={ssrc}")
                
            # Update fields
            if reception_mode is not None:
                channel.reception_mode = reception_mode
                
            if target is not None:
                az, name = self._resolve_target(target)
                channel.beam_azimuth_deg = az
                channel.beam_target = name
                
            if null_targets is not None:
                channel.null_azimuths_deg = []
                channel.null_targets = []
                for nt in null_targets:
                    az, name = self._resolve_target(nt)
                    if az is not None:
                        channel.null_azimuths_deg.append(az)
                        channel.null_targets.append(name or str(nt))
                        
            if combining_method is not None:
                channel.combining_method = combining_method
                
            logger.info(f"Reconfigured channel SSRC={ssrc}: mode={channel.reception_mode}")
            
            return channel
            
    def estimate_aoa(
        self,
        frequency_hz: float,
        duration_sec: float = 5.0,
        algorithm: str = "music",
        max_sources: int = 3,
    ) -> Dict[str, Any]:
        """
        Estimate angle-of-arrival for signals on a frequency.
        
        Args:
            frequency_hz: Frequency to analyze
            duration_sec: Analysis duration
            algorithm: AoA algorithm ("music", "esprit", "beamscan")
            max_sources: Maximum sources to detect
            
        Returns:
            AoA estimation results
        """
        # This would be implemented by the engine
        # For now, return a placeholder
        return {
            "frequency_hz": frequency_hz,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sources": [],
            "noise_floor_db": -80.0,
            "algorithm": algorithm,
        }
        
    def close(self) -> None:
        """Close the control connection."""
        if self._closed:
            return
            
        # Remove all channels
        for ssrc in list(self._channels.keys()):
            self.remove_channel(ssrc)
            
        self._closed = True
        logger.info("PhaseEngineControl closed")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
        
    def _allocate_ssrc(self) -> int:
        """Allocate a unique SSRC."""
        self._ssrc_counter += 1
        return 0x50450000 + self._ssrc_counter  # "PE" prefix
        
    def _allocate_multicast_address(self, ssrc: int) -> str:
        """Allocate a multicast address for a channel."""
        # Generate address in 239.x.x.x range based on SSRC
        b1 = (ssrc >> 16) & 0xFF
        b2 = (ssrc >> 8) & 0xFF
        b3 = ssrc & 0xFF
        return f"239.{b1}.{b2}.{b3}"
        
    def _resolve_target(
        self,
        target: Target,
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Resolve a target specification to azimuth and name.
        
        Args:
            target: Station name, azimuth, or (lat, lon)
            
        Returns:
            (azimuth_deg, name) tuple
        """
        if isinstance(target, (int, float)):
            # Direct azimuth
            return float(target), None
            
        if isinstance(target, tuple) and len(target) == 2:
            # (lat, lon) coordinates - calculate azimuth from array center
            lat, lon = target
            if self._capabilities:
                from ..sources.broadcasts import calculate_azimuth
                az = calculate_azimuth(
                    self._capabilities.array_center_lat,
                    self._capabilities.array_center_lon,
                    lat, lon
                )
                return az, f"({lat:.2f}, {lon:.2f})"
            return None, None
            
        if isinstance(target, str):
            # Station name - look up in database
            return self._lookup_station_azimuth(target), target
            
        return None, None
        
    def _lookup_station_azimuth(self, station_name: str) -> Optional[float]:
        """Look up azimuth to a known station."""
        from ..sources.broadcasts import STATIONS, get_station_azimuth
        
        # Normalize name
        name_upper = station_name.upper()
        
        # Check known stations
        station = STATIONS.get(name_upper)
        if station and self._capabilities:
            return get_station_azimuth(
                station,
                self._capabilities.array_center_lat,
                self._capabilities.array_center_lon,
            )
            
        # Could extend to external station database
        logger.warning(f"Unknown station: {station_name}")
        return None
        
    def _resolve_reception_mode(
        self,
        mode: str,
        target: Optional[Target],
        null_targets: Optional[List[Target]],
        frequency_hz: float,
    ) -> str:
        """Resolve 'auto' mode to a specific mode."""
        if mode != "auto":
            return mode
            
        # Auto mode logic
        if null_targets:
            if target:
                return "focus_null"
            return "null"
            
        if target:
            return "focus"
            
        # Check if frequency maps to known station
        from ..sources.broadcasts import get_broadcasts_for_frequency
        broadcasts = get_broadcasts_for_frequency(frequency_hz)
        
        if len(broadcasts) > 1:
            # Multiple stations on this frequency - use adaptive
            return "adaptive"
        elif len(broadcasts) == 1:
            # Single known station - focus on it
            return "focus"
            
        return "omni"
