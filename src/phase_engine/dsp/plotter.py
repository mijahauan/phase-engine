import logging
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .array_geometry import AntennaArray, calculate_bearing

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class PatternPlotter:
    """Generates theoretical radiation pattern plots for an antenna array."""

    def __init__(self, array: AntennaArray, qth_lat: float, qth_lon: float):
        self.array = array
        self.qth_lat = qth_lat
        self.qth_lon = qth_lon

    def simulate_covariance_matrix(
        self, freq_hz: float, targets: List[Tuple[float, float]], nulls: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Simulates a covariance matrix Rxx with signals arriving from specified bearings."""
        Rxx = np.zeros((self.array.n_elements, self.array.n_elements), dtype=complex)

        # Add target signals
        for lat, lon in targets:
            bearing = calculate_bearing(self.qth_lat, self.qth_lon, lat, lon)
            a = self.array.get_steering_vector(freq_hz, bearing)
            # Power of 10.0 for targets
            Rxx += 10.0 * np.outer(a, a.conj())

        # Add strong interference for nulls
        for lat, lon in nulls:
            bearing = calculate_bearing(self.qth_lat, self.qth_lon, lat, lon)
            a = self.array.get_steering_vector(freq_hz, bearing)
            # Very strong power for interferers (100.0)
            Rxx += 100.0 * np.outer(a, a.conj())

        # Add noise floor (1.0)
        Rxx += 1.0 * np.eye(self.array.n_elements)

        return Rxx

    def compute_weights(
        self,
        freq_hz: float,
        method: str,
        target_coords: List[Tuple[float, float]],
        null_coords: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Computes combining weights based on the method."""
        if not target_coords and method not in ["omni", "selection"]:
            raise ValueError("Focus method requires at least one target coordinate.")

        if method == "omni" or method == "selection":
            # Just use reference antenna (omni)
            w = np.zeros(self.array.n_elements, dtype=complex)
            w[0] = 1.0
            return w

        # For multiple targets, we synthesize a joint steering vector.
        # This is a simplification for visualization; a true multi-beam system
        # might compute separate weights per target and sum the output.
        a_target = np.zeros(self.array.n_elements, dtype=complex)
        for lat, lon in target_coords:
            bearing = calculate_bearing(self.qth_lat, self.qth_lon, lat, lon)
            a_target += self.array.get_steering_vector(freq_hz, bearing)

        if method in ["focus", "beamform", "mrc", "egc"]:
            # Simple delay-and-sum matched filter
            return a_target / np.sqrt(self.array.n_elements)

        elif method in ["adaptive", "mvdr", "focus_null", "null"]:
            Rxx = self.simulate_covariance_matrix(freq_hz, target_coords, null_coords)

            # Add diagonal loading
            Rxx_loaded = Rxx + 0.01 * np.mean(np.diag(np.abs(Rxx))) * np.eye(self.array.n_elements)

            try:
                Rxx_inv = np.linalg.inv(Rxx_loaded)
                num = Rxx_inv @ a_target
                den = a_target.conj().T @ Rxx_inv @ a_target
                w = num / den
                return w
            except np.linalg.LinAlgError:
                logger.warning("Matrix inversion failed, falling back to basic focus")
                return a_target / np.sqrt(self.array.n_elements)
        else:
            raise ValueError(f"Unknown combining method: {method}")

    def generate_plot(
        self,
        freq_hz: float,
        method: str,
        targets: List[Dict[str, Any]],
        nulls: List[Dict[str, Any]],
        output_path: str,
        title: Optional[str] = None,
    ):
        """
        Generates and saves a polar plot of the array pattern.

        Args:
            targets: List of dicts like {"name": "WWV", "lat": 40.0, "lon": -105.0}
            nulls: List of dicts like {"name": "BPM", "lat": 34.0, "lon": 109.0}
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib must be installed to generate plots.")

        target_coords = [(t["lat"], t["lon"]) for t in targets]
        null_coords = [(n["lat"], n["lon"]) for n in nulls]

        weights = self.compute_weights(freq_hz, method, target_coords, null_coords)

        angles = np.linspace(0, 360, 360)
        response = np.zeros_like(angles, dtype=complex)

        for i, angle in enumerate(angles):
            a_theta = self.array.get_steering_vector(freq_hz, angle)
            response[i] = weights.conj().T @ a_theta

        power = np.abs(response) ** 2
        power_db = 10 * np.log10(power + 1e-10)
        power_db = power_db - np.max(power_db)  # Normalize peak to 0dB
        power_db = np.maximum(power_db, -40)  # Floor at -40dB

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        theta_rad = np.radians(angles)
        ax.plot(theta_rad, power_db, linewidth=2, color="blue")
        ax.fill(theta_rad, power_db, alpha=0.2, color="blue")

        # Plot target markers
        for t in targets:
            bearing = calculate_bearing(self.qth_lat, self.qth_lon, t["lat"], t["lon"])
            ax.plot(
                math.radians(bearing),
                0,
                "go",
                markersize=10,
                label=f'Focus: {t.get("name", "Target")}',
            )

        # Plot null markers
        for n in nulls:
            bearing = calculate_bearing(self.qth_lat, self.qth_lon, n["lat"], n["lon"])
            ax.plot(
                math.radians(bearing),
                0,
                "ro",
                markersize=10,
                label=f'Null: {n.get("name", "Interferer")}',
            )

        ax.set_rlabel_position(-22.5)
        ax.set_rticks([-40, -30, -20, -10, 0])

        if not title:
            target_names = "+".join([t.get("name", "T") for t in targets])
            title = f"Pattern: {method.upper()} @ {freq_hz/1e6:.2f} MHz"
            if target_names:
                title += f" [Targets: {target_names}]"

        ax.set_title(title, va="bottom", pad=20)

        # Only add legend if we plotted markers
        if targets or nulls:
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Saved pattern plot to {output_path}")
