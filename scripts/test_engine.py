#!/usr/bin/env python3
"""
Test script for PhaseEngine with three radiod sources.

Tests:
1. Connect to bee1, bee3, bee4
2. Calibrate using 900 kHz AM station
3. Create channels for all 9 broadcast frequencies
4. Capture and combine samples for all 17 broadcasts
"""

import sys
import time
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from ka9q
logging.getLogger('ka9q').setLevel(logging.WARNING)

from phase_engine import PhaseEngine, SourceConfig
from phase_engine.sources import BROADCASTS, FREQUENCIES_HZ


def main():
    # Kansas City QTH
    QTH_LAT = 39.0997
    QTH_LON = -94.5786
    
    # Source configurations
    sources = [
        SourceConfig(
            name="bee1",
            status_address="bee1-status.local",
            position=(0.0, 0.0, 0.0),  # Reference antenna at origin
        ),
        SourceConfig(
            name="bee3",
            status_address="bee3-status.local",
            position=(10.0, 0.0, 0.0),  # 10m East
        ),
        SourceConfig(
            name="bee4",
            status_address="bee4-status.local",
            position=(0.0, 10.0, 0.0),  # 10m North
        ),
    ]
    
    print("=" * 60)
    print("PhaseEngine Test")
    print("=" * 60)
    print()
    
    # Create engine
    engine = PhaseEngine(
        qth_latitude=QTH_LAT,
        qth_longitude=QTH_LON,
        sources=sources,
        reference_source="bee1",
        sample_rate=12000,
        calibration_frequency_hz=900e3,  # Strong local AM station
    )
    
    try:
        # Connect
        print("Connecting to radiod sources...")
        engine.connect()
        print()
        
        # Calibrate
        print("Calibrating...")
        cal = engine.calibrate(duration_sec=3.0)
        print()
        print("Calibration Results:")
        print(f"  Reference: {cal.reference_source}")
        for name in sorted(cal.source_delays.keys()):
            delay = cal.source_delays[name]
            phase = cal.source_phases[name]
            corr = cal.correlation_coefficients[name]
            print(f"  {name}: delay={delay:4d} samples, phase={phase:7.2f}°, corr={corr:.4f}")
        print()
        
        # Create broadcast channels
        print(f"Creating channels for {len(FREQUENCIES_HZ)} frequencies...")
        engine.create_broadcast_channels()
        print()
        
        # Start capture
        print("Starting capture...")
        engine.start()
        
        # Wait for samples
        print("Capturing for 3 seconds...")
        time.sleep(3.5)
        
        # Get combined samples for all broadcasts
        print()
        print("Combined Broadcast Results:")
        print("-" * 60)
        
        combined = engine.get_all_combined_samples()
        
        for broadcast in BROADCASTS:
            samples = combined.get(broadcast)
            if samples is not None and len(samples) > 0:
                power = np.mean(np.abs(samples) ** 2)
                power_db = 10 * np.log10(power) if power > 0 else -999
                print(f"  {broadcast.call_sign:5s} {broadcast.frequency_mhz:6.3f} MHz: "
                      f"{len(samples):6d} samples, power={power_db:+6.1f} dB")
            else:
                print(f"  {broadcast.call_sign:5s} {broadcast.frequency_mhz:6.3f} MHz: NO DATA")
                
        print()
        
        # Stop
        engine.stop()
        
        # Show status
        print("Final Status:")
        status = engine.get_status()
        print(f"  Running: {status['running']}")
        print(f"  Calibrated: {status['calibrated']}")
        for name, src_status in status['sources'].items():
            print(f"  {name}: connected={src_status['connected']}, "
                  f"channels={len(src_status['frequencies'])}")
                  
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1
        
    finally:
        print()
        print("Disconnecting...")
        engine.disconnect()
        
    print()
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
