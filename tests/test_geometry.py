import pytest
import numpy as np
import math

from phase_engine.dsp.array_geometry import AntennaArray
from phase_engine.engine import calculate_bearing

def test_calculate_bearing():
    # Test North
    assert pytest.approx(calculate_bearing(0, 0, 10, 0), 0.1) == 0.0
    # Test East
    assert pytest.approx(calculate_bearing(0, 0, 0, 10), 0.1) == 90.0
    # Test South
    assert pytest.approx(calculate_bearing(10, 0, 0, 0), 0.1) == 180.0
    # Test West
    assert pytest.approx(calculate_bearing(0, 10, 0, 0), 0.1) == 270.0

def test_antenna_array_initialization():
    positions = {
        "center": (0.0, 0.0, 0.0),
        "north": (0.0, 10.0, 0.0),
        "east": (10.0, 0.0, 0.0)
    }
    
    array = AntennaArray("center", positions)
    assert array.n_elements == 3
    assert array.antenna_names == ["center", "north", "east"]
    
    # Check coordinates
    assert np.allclose(array.coords[:, 0], [0.0, 0.0, 0.0]) # Center
    assert np.allclose(array.coords[:, 1], [0.0, 10.0, 0.0]) # North
    assert np.allclose(array.coords[:, 2], [10.0, 0.0, 0.0]) # East

def test_steering_vector_broadside():
    # 10 MHz
    freq = 10e6
    wavelength = 299792458.0 / freq
    
    # Signal coming from North (0 degrees azimuth)
    positions = {
        "center": (0.0, 0.0, 0.0),
        "north": (0.0, wavelength / 2, 0.0) # Exactly half a wavelength North
    }
    array = AntennaArray("center", positions)
    
    sv = array.get_steering_vector(freq, azimuth_deg=0.0)
    
    # Center should always be 1 + 0j (0 phase shift)
    assert np.allclose(sv[0], 1.0 + 0j)
    
    # North should be phase shifted by 180 degrees (-1 + 0j)
    assert np.allclose(sv[1], -1.0 + 0j, atol=1e-5)
