import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
import math

logger = logging.getLogger(__name__)

# Speed of light in vacuum (m/s)
C_SPEED = 299792458.0


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing from point 1 to point 2 in degrees."""
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    initial_bearing = math.atan2(y, x)
    return (math.degrees(initial_bearing) + 360) % 360


class AntennaArray:
    """
    Models the physical geometry and behavior of the antenna array.
    """

    def __init__(self, reference_name: str, positions: Dict[str, Tuple[float, float, float]]):
        """
        Initialize the array geometry.

        Args:
            reference_name: Name of the reference antenna (phase center)
            positions: Dict mapping antenna name to (x, y, z) coordinates in meters.
                       Typically ENU (East, North, Up) relative to the reference.
        """
        self.reference_name = reference_name
        self.positions = positions

        if reference_name not in positions:
            raise ValueError(f"Reference antenna '{reference_name}' not in positions dict")

        self.antenna_names = list(positions.keys())
        self.n_elements = len(self.antenna_names)

        # Build coordinate matrix [3 x N]
        self.coords = np.zeros((3, self.n_elements))

        # Let's shift everything so the reference is strictly at (0,0,0)
        ref_pos = np.array(positions[reference_name])

        for i, name in enumerate(self.antenna_names):
            self.coords[:, i] = np.array(positions[name]) - ref_pos

        logger.info(f"Initialized AntennaArray with {self.n_elements} elements")
        for i, name in enumerate(self.antenna_names):
            logger.debug(f"  {name}: {self.coords[:, i]} m")

    def get_steering_vector(
        self, frequency_hz: float, azimuth_deg: float, elevation_deg: float = 0.0
    ) -> np.ndarray:
        """
        Calculate the steering vector for a plane wave arriving from a specific direction.

        Args:
            frequency_hz: Signal frequency in Hz
            azimuth_deg: Arrival azimuth in degrees (0 = North, 90 = East)
            elevation_deg: Arrival elevation in degrees (0 = Horizon, 90 = Zenith)

        Returns:
            np.ndarray of shape (N,) containing complex weights to phase-align the array
        """
        wavelength = C_SPEED / frequency_hz
        k = 2 * np.pi / wavelength

        # Convert to radians
        az_rad = math.radians(azimuth_deg)
        el_rad = math.radians(elevation_deg)

        # Wave number vector (direction of ARRIVAL)
        # Using standard radar convention where signal arrives FROM (az, el)
        # Assuming coordinates are ENU (x=East, y=North, z=Up)
        # 0 deg Azimuth = North (y-axis)
        # 90 deg Azimuth = East (x-axis)

        kx = -math.sin(az_rad) * math.cos(el_rad)
        ky = -math.cos(az_rad) * math.cos(el_rad)
        kz = -math.sin(el_rad)

        k_vec = np.array([kx, ky, kz]) * k

        # Calculate phase delays: phase = k_vec dot position
        phases = np.dot(k_vec, self.coords)

        # Steering vector is exp(-j * phase)
        # (Assuming narrow-band model where time delay = phase shift)
        a = np.exp(-1j * phases)

        return a

    def get_spatial_covariance_matrix(self, samples_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the spatial covariance matrix of the array snapshots.

        Args:
            samples_matrix: Array of shape (N_antennas, N_samples)

        Returns:
            Rxx: Spatial covariance matrix of shape (N_antennas, N_antennas)
        """
        n_samples = samples_matrix.shape[1]
        # Rxx = E[x * x^H]
        Rxx = (samples_matrix @ samples_matrix.conj().T) / n_samples
        return Rxx
