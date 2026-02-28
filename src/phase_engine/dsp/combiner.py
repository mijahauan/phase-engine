import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .array_geometry import AntennaArray

logger = logging.getLogger(__name__)

@dataclass
class CombiningWeights:
    """Store the complex weights used for combining."""
    weights: np.ndarray  # Shape: (N_antennas,)
    method: str
    target_name: Optional[str] = None
    snr_estimate_db: float = 0.0

class PhaseCombiner:
    """
    Implements various array signal processing algorithms (MRC, EGC, Selection, MVDR)
    to combine multi-antenna samples into a single stream.
    """
    def __init__(self, array: AntennaArray):
        self.array = array
        # Optional calibration corrections (complex multipliers per antenna)
        self.calibration_weights = np.ones(self.array.n_elements, dtype=np.complex64)
        
    def set_calibration(self, weights: np.ndarray):
        """Set base calibration weights (e.g. from static reference source)."""
        if len(weights) != self.array.n_elements:
            raise ValueError("Calibration weights must match number of antennas")
        self.calibration_weights = weights

    def _apply_calibration(self, samples_matrix: np.ndarray) -> np.ndarray:
        """Apply static calibration before adaptive processing."""
        # Multiply each row (antenna) by its calibration weight
        return samples_matrix * self.calibration_weights[:, np.newaxis]

    def _map_samples_to_matrix(self, samples_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert dict of samples to an aligned (N_antennas, N_samples) matrix.
        Returns the matrix and the list of antenna names that were present.
        """
        available_names = []
        for name in self.array.antenna_names:
            if name in samples_dict and samples_dict[name] is not None:
                available_names.append(name)
                
        if not available_names:
            return np.array([]), []
            
        # Get minimum length to handle slight buffering mismatches
        min_len = min(len(samples_dict[name]) for name in available_names)
        if min_len == 0:
            return np.array([]), available_names
            
        matrix = np.zeros((len(available_names), min_len), dtype=np.complex64)
        for i, name in enumerate(available_names):
            matrix[i, :] = samples_dict[name][:min_len]
            
        return matrix, available_names

    def process(
        self, 
        samples_dict: Dict[str, np.ndarray], 
        method: str = "mrc",
        steering_vector: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Combine samples using the requested method.
        
        Args:
            samples_dict: Map of antenna name to complex samples
            method: 'mrc', 'egc', 'selection', 'beamform', 'mvdr'
            steering_vector: Required for 'beamform' and 'mvdr'
            
        Returns:
            Combined complex samples array (1D)
        """
        raw_matrix, names = self._map_samples_to_matrix(samples_dict)
        if raw_matrix.size == 0:
            return None
            
        # If we have missing antennas, we need to adapt our processing
        # For now, let's assume all antennas are present.
        # TODO: Handle partial array subsets robustly
        if len(names) != self.array.n_elements:
            # Fallback to selection diversity (use first available)
            logger.debug("Missing antennas, falling back to simple selection")
            return raw_matrix[0, :]
            
        # Apply static phase/amplitude calibration
        X = self._apply_calibration(raw_matrix)
        
        weights = None
        
        if method == "selection":
            weights = self._calc_selection_weights(X)
        elif method == "egc":
            weights = self._calc_egc_weights(X)
        elif method == "mrc":
            weights = self._calc_mrc_weights(X)
        elif method == "beamform":
            if steering_vector is None:
                logger.warning("Beamforming requested but no steering vector provided, using EGC")
                weights = self._calc_egc_weights(X)
            else:
                weights = self._calc_beamform_weights(steering_vector)
        elif method == "mvdr":
            if steering_vector is None:
                logger.warning("MVDR requested but no steering vector provided, using MRC")
                weights = self._calc_mrc_weights(X)
            else:
                weights = self._calc_mvdr_weights(X, steering_vector)
        else:
            logger.warning(f"Unknown combining method {method}, falling back to MRC")
            weights = self._calc_mrc_weights(X)
            
        # Apply weights: y = w^H * X
        # where w^H is conjugate transpose. So we conjugate weights.
        combined = np.dot(weights.conj(), X)
        return combined
        
    def _calc_selection_weights(self, X: np.ndarray) -> np.ndarray:
        """Selection Diversity: Pick antenna with highest average power."""
        powers = np.mean(np.abs(X)**2, axis=1)
        best_idx = np.argmax(powers)
        w = np.zeros(self.array.n_elements, dtype=np.complex64)
        w[best_idx] = 1.0
        return w
        
    def _calc_egc_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Equal Gain Combining (EGC): Co-phase signals but ignore amplitude differences.
        Uses the first element as phase reference.
        """
        # Calculate cross-correlation with reference element
        ref_signal = X[0, :]
        cross_corr = np.mean(X * ref_signal.conj(), axis=1)
        
        # We want to multiply by the phase that cancels this difference
        phases = np.angle(cross_corr)
        
        # EGC sets magnitude to 1
        w = np.exp(1j * phases)
        # Normalize to prevent power explosion
        w = w / np.sqrt(self.array.n_elements)
        return w
        
    def _calc_mrc_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Maximum Ratio Combining (MRC): Weight by both phase alignment and SNR.
        In noise-limited environments, MRC is optimal.
        """
        ref_signal = X[0, :]
        cross_corr = np.mean(X * ref_signal.conj(), axis=1)
        
        # Weight vector is proportional to the channel estimate (cross correlation)
        # w = h
        w = cross_corr
        
        # Normalize power to preserve unit gain
        norm = np.linalg.norm(w)
        if norm > 0:
            w = w / norm
        else:
            w = self._calc_selection_weights(X)
            
        return w
        
    def _calc_beamform_weights(self, steering_vector: np.ndarray) -> np.ndarray:
        """
        Delay-and-Sum Beamformer (Spatial Matched Filter).
        Just uses the steering vector.
        """
        w = steering_vector
        w = w / np.sqrt(self.array.n_elements)
        return w
        
    def _calc_mvdr_weights(self, X: np.ndarray, steering_vector: np.ndarray) -> np.ndarray:
        """
        Minimum Variance Distortionless Response (MVDR / Capon Beamformer).
        Minimizes total array output power while maintaining unity gain toward the target.
        Effectively places nulls at interference sources.
        """
        Rxx = self.array.get_spatial_covariance_matrix(X)
        
        # Add diagonal loading for numerical stability (prevent matrix inversion blowup)
        # Load with 1% of the mean diagonal power
        diag_load = 0.01 * np.mean(np.diag(np.abs(Rxx)))
        Rxx_loaded = Rxx + np.eye(self.array.n_elements) * diag_load
        
        try:
            Rxx_inv = np.linalg.inv(Rxx_loaded)
            
            # w = (Rxx^-1 * a) / (a^H * Rxx^-1 * a)
            a = steering_vector
            num = Rxx_inv @ a
            den = a.conj().T @ Rxx_inv @ a
            
            w = num / den
            return w
        except np.linalg.LinAlgError:
            logger.warning("MVDR matrix inversion failed, falling back to delay-and-sum beamformer")
            return self._calc_beamform_weights(steering_vector)
