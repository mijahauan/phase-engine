import pytest
import numpy as np

from phase_engine.dsp.array_geometry import AntennaArray
from phase_engine.dsp.combiner import PhaseCombiner

def test_selection_combiner():
    positions = {"ant1": (0,0,0), "ant2": (10,0,0)}
    array = AntennaArray("ant1", positions)
    combiner = PhaseCombiner(array)
    
    # ant2 has stronger signal
    samples = {
        "ant1": np.array([0.1j, 0.1j, 0.1j]),
        "ant2": np.array([1.0j, 1.0j, 1.0j])
    }
    
    combined = combiner.process(samples, method="selection")
    
    # Should equal ant2 exactly
    assert np.allclose(combined, samples["ant2"])

def test_egc_combiner():
    positions = {"ant1": (0,0,0), "ant2": (10,0,0)}
    array = AntennaArray("ant1", positions)
    combiner = PhaseCombiner(array)
    
    # Same amplitude, 180 degrees out of phase
    samples = {
        "ant1": np.array([1.0+0j, 1.0+0j]),
        "ant2": np.array([-1.0+0j, -1.0+0j])
    }
    
    combined = combiner.process(samples, method="egc")
    
    # EGC should rotate ant2 by 180 degrees and sum them, then normalize
    # Sum = 2.0, norm = sqrt(2), so result should be ~1.414
    expected_mag = 2.0 / np.sqrt(2)
    assert np.allclose(np.abs(combined), expected_mag)
