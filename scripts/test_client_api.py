#!/usr/bin/env python3
"""
Test script for PhaseEngineControl and PhaseEngineStream client API.

Demonstrates:
1. API compatibility with ka9q-python patterns
2. Extended parameters for spatial filtering
3. Capability detection
4. Channel creation with reception modes
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from phase_engine import (
    PhaseEngineControl,
    PhaseEngineStream,
    ChannelInfo,
    Capabilities,
)
from phase_engine.sources import BROADCASTS, STATIONS, get_station_azimuth


def test_capabilities():
    """Test capability query."""
    print("=" * 60)
    print("Test: Capability Query")
    print("=" * 60)

    control = PhaseEngineControl("phase-engine-status.local")
    caps = control.get_capabilities()

    print(f"Backend: {caps['backend']}")
    print(f"Version: {caps['version']}")
    print(f"Antennas: {caps['n_antennas']}")
    print(f"DoF: {caps['dof']}")
    print(f"Modes: {caps['modes']}")
    print(f"Max nulls: {caps['max_nulls']}")
    print(f"Can focus+null: {caps['can_focus_and_null']}")

    control.close()
    print("✓ Capability query passed\n")


def test_standard_channel():
    """Test standard channel creation (RadiodControl compatible)."""
    print("=" * 60)
    print("Test: Standard Channel Creation")
    print("=" * 60)

    control = PhaseEngineControl("phase-engine-status.local")

    # Standard ka9q-python style channel creation
    channel = control.create_channel(
        frequency_hz=10e6,
        preset="iq",
        sample_rate=12000,
    )

    print(f"SSRC: {channel.ssrc:#x}")
    print(f"Frequency: {channel.frequency/1e6:.3f} MHz")
    print(f"Multicast: {channel.multicast_address}:{channel.port}")
    print(f"Sample rate: {channel.sample_rate}")
    print(f"Preset: {channel.preset}")
    print(f"Reception mode: {channel.reception_mode}")

    control.close()
    print("✓ Standard channel creation passed\n")


def test_extended_channel():
    """Test extended channel creation with spatial filtering."""
    print("=" * 60)
    print("Test: Extended Channel Creation (Spatial Filtering)")
    print("=" * 60)

    control = PhaseEngineControl("phase-engine-status.local")

    # Extended channel with beam steering
    channel = control.create_channel(
        frequency_hz=10e6,
        preset="iq",
        sample_rate=12000,
        reception_mode="focus",
        target="WWV",
        combining_method="mrc",
    )

    print(f"SSRC: {channel.ssrc:#x}")
    print(f"Frequency: {channel.frequency/1e6:.3f} MHz")
    print(f"Reception mode: {channel.reception_mode}")
    print(f"Beam target: {channel.beam_target}")
    print(
        f"Beam azimuth: {channel.beam_azimuth_deg}°"
        if channel.beam_azimuth_deg
        else "Beam azimuth: N/A"
    )
    print(f"Combining method: {channel.combining_method}")

    control.close()
    print("✓ Extended channel creation passed\n")


def test_null_channel():
    """Test channel with null steering."""
    print("=" * 60)
    print("Test: Channel with Null Steering")
    print("=" * 60)

    control = PhaseEngineControl("phase-engine-status.local")

    # Channel with focus and null
    channel = control.create_channel(
        frequency_hz=10e6,
        preset="iq",
        reception_mode="focus_null",
        target="WWV",
        null_targets=["BPM", "WWVH"],
    )

    print(f"Reception mode: {channel.reception_mode}")
    print(f"Beam target: {channel.beam_target}")
    print(f"Null targets: {channel.null_targets}")
    print(f"Null azimuths: {channel.null_azimuths_deg}")

    control.close()
    print("✓ Null steering passed\n")


def test_azimuth_target():
    """Test channel with direct azimuth target."""
    print("=" * 60)
    print("Test: Direct Azimuth Target")
    print("=" * 60)

    control = PhaseEngineControl("phase-engine-status.local")

    # Target by azimuth (degrees)
    channel = control.create_channel(
        frequency_hz=9650e3,
        preset="am",
        reception_mode="focus",
        target=265.0,  # Direct azimuth
    )

    print(f"Frequency: {channel.frequency/1e6:.3f} MHz")
    print(f"Beam azimuth: {channel.beam_azimuth_deg}°")
    print(f"Beam target: {channel.beam_target}")  # Should be None for direct azimuth

    control.close()
    print("✓ Direct azimuth target passed\n")


def test_ensure_channel():
    """Test ensure_channel (idempotent channel creation)."""
    print("=" * 60)
    print("Test: Ensure Channel (Idempotent)")
    print("=" * 60)

    control = PhaseEngineControl("phase-engine-status.local")

    # First call creates channel
    ch1 = control.ensure_channel(frequency_hz=5e6, preset="iq")
    print(f"First call: SSRC={ch1.ssrc:#x}")

    # Second call returns same channel
    ch2 = control.ensure_channel(frequency_hz=5e6, preset="iq")
    print(f"Second call: SSRC={ch2.ssrc:#x}")

    assert ch1.ssrc == ch2.ssrc, "ensure_channel should return same channel"

    control.close()
    print("✓ Ensure channel passed\n")


def test_reconfigure():
    """Test channel reconfiguration."""
    print("=" * 60)
    print("Test: Channel Reconfiguration")
    print("=" * 60)

    control = PhaseEngineControl("phase-engine-status.local")

    # Create channel in omni mode
    channel = control.create_channel(
        frequency_hz=10e6,
        preset="iq",
        reception_mode="omni",
    )
    print(f"Initial mode: {channel.reception_mode}")

    # Reconfigure to focus mode
    channel = control.reconfigure_channel(
        ssrc=channel.ssrc,
        reception_mode="focus",
        target="WWV",
    )
    print(f"After reconfigure: mode={channel.reception_mode}, target={channel.beam_target}")

    control.close()
    print("✓ Reconfiguration passed\n")


def test_auto_mode():
    """Test auto mode resolution."""
    print("=" * 60)
    print("Test: Auto Mode Resolution")
    print("=" * 60)

    control = PhaseEngineControl("phase-engine-status.local")

    # 10 MHz has multiple stations (WWV, WWVH, BPM) -> should use adaptive
    ch_multi = control.create_channel(
        frequency_hz=10e6,
        preset="iq",
        reception_mode="auto",
    )
    print(f"10 MHz (multi-station): mode={ch_multi.reception_mode}")

    # 20 MHz has only WWV -> should use focus
    ch_single = control.create_channel(
        frequency_hz=20e6,
        preset="iq",
        reception_mode="auto",
    )
    print(f"20 MHz (single station): mode={ch_single.reception_mode}")

    # 9.65 MHz has no known station -> should use omni
    ch_unknown = control.create_channel(
        frequency_hz=9.65e6,
        preset="iq",
        reception_mode="auto",
    )
    print(f"9.65 MHz (unknown): mode={ch_unknown.reception_mode}")

    control.close()
    print("✓ Auto mode resolution passed\n")


def test_context_manager():
    """Test context manager usage."""
    print("=" * 60)
    print("Test: Context Manager")
    print("=" * 60)

    with PhaseEngineControl("phase-engine-status.local") as control:
        channel = control.create_channel(frequency_hz=15e6, preset="iq")
        print(f"Created channel: {channel.frequency/1e6:.3f} MHz")

    print("✓ Context manager passed\n")


def main():
    print("\n" + "=" * 60)
    print("PhaseEngineControl Client API Tests")
    print("=" * 60 + "\n")

    try:
        test_capabilities()
        test_standard_channel()
        test_extended_channel()
        test_null_channel()
        test_azimuth_target()
        test_ensure_channel()
        test_reconfigure()
        test_auto_mode()
        test_context_manager()

        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
