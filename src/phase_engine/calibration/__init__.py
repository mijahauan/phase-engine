"""Calibration - Startup routines for sample alignment and phase offset."""

from .sample_align import (
    SampleAligner,
    CalibrationResult,
    run_calibration_routine,
)

from .sources import (
    AMStation,
    AMStationDatabase,
    CalibrationSourceSet,
    find_calibration_sources,
    CLEAR_CHANNEL_FREQUENCIES_KHZ,
)

from .terrestrial import (
    TerrestrialCalibrator,
    TerrestrialCalibrationResult,
    ArrayCalibrationResult,
    predict_phase_from_geometry,
    run_terrestrial_calibration,
)

__all__ = [
    # Sample alignment (wideband cross-correlation)
    'SampleAligner',
    'CalibrationResult',
    'run_calibration_routine',
    
    # AM station sources
    'AMStation',
    'AMStationDatabase', 
    'CalibrationSourceSet',
    'find_calibration_sources',
    'CLEAR_CHANNEL_FREQUENCIES_KHZ',
    
    # Terrestrial (geometric) calibration
    'TerrestrialCalibrator',
    'TerrestrialCalibrationResult',
    'ArrayCalibrationResult',
    'predict_phase_from_geometry',
    'run_terrestrial_calibration',
]
