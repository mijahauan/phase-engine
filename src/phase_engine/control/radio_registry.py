"""
Radio Registry - Manage multiple physical radios as named entities.

Allows the phase-engine to command "Antenna_North" and "Antenna_South"
individually while presenting a unified interface upstream.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class AntennaRole(Enum):
    """Role of an antenna in the array geometry."""
    REFERENCE = "reference"  # Phase reference (typically North or center)
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    AUXILIARY = "auxiliary"


@dataclass
class AntennaConfig:
    """Configuration for a single antenna/radio in the array."""
    name: str
    role: AntennaRole
    status_address: str  # ka9q-radio status multicast address
    data_address: str    # RTP data multicast address
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) in meters
    calibration_delay_samples: int = 0
    calibration_phase_deg: float = 0.0
    enabled: bool = True
    
    # Runtime state
    connected: bool = field(default=False, repr=False)
    last_seen: Optional[float] = field(default=None, repr=False)


class RadioRegistry:
    """
    Registry of physical radios in the phased array.
    
    Provides named access to radios and handles the mapping between
    logical antenna names and physical ka9q-radio instances.
    """
    
    def __init__(self):
        self._radios: Dict[str, AntennaConfig] = {}
        self._reference: Optional[str] = None
    
    def register(self, config: AntennaConfig) -> None:
        """Register an antenna/radio in the array."""
        if config.name in self._radios:
            logger.warning(f"Overwriting existing radio: {config.name}")
        
        self._radios[config.name] = config
        logger.info(f"Registered radio: {config.name} ({config.role.value})")
        
        # Auto-set reference if this is the first or marked as reference
        if config.role == AntennaRole.REFERENCE or self._reference is None:
            self._reference = config.name
            logger.info(f"Set phase reference: {config.name}")
    
    def unregister(self, name: str) -> None:
        """Remove an antenna/radio from the registry."""
        if name in self._radios:
            del self._radios[name]
            if self._reference == name:
                self._reference = next(iter(self._radios), None)
            logger.info(f"Unregistered radio: {name}")
    
    def get(self, name: str) -> Optional[AntennaConfig]:
        """Get antenna configuration by name."""
        return self._radios.get(name)
    
    def get_reference(self) -> Optional[AntennaConfig]:
        """Get the phase reference antenna."""
        if self._reference:
            return self._radios.get(self._reference)
        return None
    
    def get_enabled(self) -> Dict[str, AntennaConfig]:
        """Get all enabled antennas."""
        return {name: cfg for name, cfg in self._radios.items() if cfg.enabled}
    
    def get_all(self) -> Dict[str, AntennaConfig]:
        """Get all registered antennas."""
        return dict(self._radios)
    
    @property
    def count(self) -> int:
        """Number of registered radios."""
        return len(self._radios)
    
    @property
    def enabled_count(self) -> int:
        """Number of enabled radios."""
        return sum(1 for cfg in self._radios.values() if cfg.enabled)
    
    @property
    def degrees_of_freedom(self) -> int:
        """Degrees of freedom for beamforming (N-1)."""
        return max(0, self.enabled_count - 1)
    
    def get_baselines(self) -> list[tuple[str, str, float]]:
        """
        Calculate all baseline distances between antenna pairs.
        
        Returns:
            List of (antenna_a, antenna_b, distance_m) tuples
        """
        import math
        baselines = []
        names = list(self.get_enabled().keys())
        
        for i, name_a in enumerate(names):
            for name_b in names[i+1:]:
                cfg_a = self._radios[name_a]
                cfg_b = self._radios[name_b]
                
                dx = cfg_b.position_m[0] - cfg_a.position_m[0]
                dy = cfg_b.position_m[1] - cfg_a.position_m[1]
                dz = cfg_b.position_m[2] - cfg_a.position_m[2]
                
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                baselines.append((name_a, name_b, distance))
        
        return baselines
    
    def update_calibration(self, name: str, delay_samples: int, phase_deg: float) -> None:
        """Update calibration constants for an antenna."""
        if name in self._radios:
            self._radios[name].calibration_delay_samples = delay_samples
            self._radios[name].calibration_phase_deg = phase_deg
            logger.info(f"Updated calibration for {name}: "
                       f"delay={delay_samples} samples, phase={phase_deg:.1f}°")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry to dictionary."""
        return {
            'reference': self._reference,
            'radios': {
                name: {
                    'role': cfg.role.value,
                    'status_address': cfg.status_address,
                    'data_address': cfg.data_address,
                    'position_m': cfg.position_m,
                    'calibration_delay_samples': cfg.calibration_delay_samples,
                    'calibration_phase_deg': cfg.calibration_phase_deg,
                    'enabled': cfg.enabled,
                }
                for name, cfg in self._radios.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RadioRegistry':
        """Deserialize registry from dictionary."""
        registry = cls()
        
        for name, cfg_data in data.get('radios', {}).items():
            config = AntennaConfig(
                name=name,
                role=AntennaRole(cfg_data['role']),
                status_address=cfg_data['status_address'],
                data_address=cfg_data['data_address'],
                position_m=tuple(cfg_data.get('position_m', (0, 0, 0))),
                calibration_delay_samples=cfg_data.get('calibration_delay_samples', 0),
                calibration_phase_deg=cfg_data.get('calibration_phase_deg', 0.0),
                enabled=cfg_data.get('enabled', True),
            )
            registry.register(config)
        
        # Restore reference if specified
        if data.get('reference') in registry._radios:
            registry._reference = data['reference']
        
        return registry
