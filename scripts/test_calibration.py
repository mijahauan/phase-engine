#!/usr/bin/env python3
"""
Test script for phase-engine calibration with two radiod instances.

This script:
1. Connects to bee1-status.local and bee4-status.local
2. Creates channels on the same AM frequency
3. Captures samples simultaneously from both
4. Runs sample alignment (Stage 1)
5. Runs terrestrial phase calibration (Stage 2)

Usage:
    python scripts/test_calibration.py

Requirements:
    - ka9q-python >= 3.3.0
    - Two radiod instances running (bee1, bee4)
    - Network connectivity to both
"""

import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ka9q import RadiodControl, RadiodStream, discover_channels, ChannelInfo

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RadiodInstance:
    """Configuration for a single radiod instance."""

    name: str
    status_address: str
    control: Optional[RadiodControl] = None


class MultiRadiodCapture:
    """
    Capture synchronized samples from multiple radiod instances.

    Each radiod is tuned to the same frequency, and samples are captured
    simultaneously for cross-correlation and calibration.
    """

    def __init__(self, instances: list[RadiodInstance]):
        """
        Initialize multi-radiod capture.

        Args:
            instances: List of RadiodInstance configurations
        """
        self.instances = {inst.name: inst for inst in instances}
        self._connected = False

    def connect(self) -> bool:
        """Connect to all radiod instances."""
        logger.info(f"Connecting to {len(self.instances)} radiod instances...")

        for name, inst in self.instances.items():
            try:
                inst.control = RadiodControl(inst.status_address)
                logger.info(f"  {name}: Connected to {inst.status_address}")
            except Exception as e:
                logger.error(f"  {name}: Failed to connect: {e}")
                return False

        self._connected = True
        return True

    def disconnect(self):
        """Disconnect from all radiod instances."""
        for name, inst in self.instances.items():
            if inst.control:
                try:
                    inst.control.close()
                except Exception:
                    pass
                inst.control = None
        self._connected = False

    def tune_all(
        self,
        frequency_hz: float,
        preset: str = "am",
        sample_rate: int = 12000,
    ) -> Dict[str, ChannelInfo]:
        """
        Tune all radiod instances to the same frequency.

        Args:
            frequency_hz: Center frequency in Hz
            preset: Demodulation preset (am, iq, usb, etc.)
            sample_rate: Output sample rate

        Returns:
            Dict of instance name -> ChannelInfo
        """
        if not self._connected:
            raise RuntimeError("Not connected to radiod instances")

        channels = {}

        for name, inst in self.instances.items():
            try:
                channel = inst.control.ensure_channel(
                    frequency_hz=frequency_hz,
                    preset=preset,
                    sample_rate=sample_rate,
                )
                channels[name] = channel
                logger.info(
                    f"  {name}: Tuned to {frequency_hz/1e6:.3f} MHz, " f"SSRC={channel.ssrc}"
                )
            except Exception as e:
                logger.error(f"  {name}: Failed to tune: {e}")
                raise

        return channels

    def capture_samples(
        self,
        channels: Dict[str, ChannelInfo],
        duration_s: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Capture samples from all channels simultaneously.

        Args:
            channels: Dict of instance name -> ChannelInfo
            duration_s: Capture duration in seconds

        Returns:
            Dict of instance name -> complex sample array
        """
        samples = {}
        sample_buffers = {name: [] for name in channels}
        capture_complete = {name: False for name in channels}

        def capture_one(name: str, channel: ChannelInfo):
            """Capture from a single channel."""
            buffer = []
            samples_needed = int(duration_s * channel.sample_rate)
            samples_captured = 0

            def on_samples(samps: np.ndarray, quality):
                nonlocal samples_captured
                buffer.append(samps.copy())
                samples_captured += len(samps)

            stream = RadiodStream(
                channel=channel,
                on_samples=on_samples,
            )

            stream.start()

            # Wait for enough samples
            start_time = time.time()
            timeout = duration_s + 5.0  # Extra time for startup

            while samples_captured < samples_needed:
                if time.time() - start_time > timeout:
                    logger.warning(
                        f"{name}: Capture timeout, got {samples_captured}/{samples_needed}"
                    )
                    break
                time.sleep(0.1)

            stream.stop()

            # Concatenate all captured samples
            if buffer:
                return np.concatenate(buffer)[:samples_needed]
            return np.array([], dtype=np.complex64)

        # Capture from all channels in parallel
        logger.info(f"Capturing {duration_s}s from {len(channels)} channels...")

        with ThreadPoolExecutor(max_workers=len(channels)) as executor:
            futures = {
                executor.submit(capture_one, name, channel): name
                for name, channel in channels.items()
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    samples[name] = future.result()
                    logger.info(f"  {name}: Captured {len(samples[name])} samples")
                except Exception as e:
                    logger.error(f"  {name}: Capture failed: {e}")
                    samples[name] = np.array([], dtype=np.complex64)

        return samples

    def remove_channels(self, channels: Dict[str, ChannelInfo]):
        """Remove channels from all radiod instances."""
        for name, channel in channels.items():
            inst = self.instances.get(name)
            if inst and inst.control:
                try:
                    inst.control.remove_channel(channel.ssrc)
                    logger.debug(f"  {name}: Removed channel SSRC={channel.ssrc}")
                except Exception as e:
                    logger.warning(f"  {name}: Failed to remove channel: {e}")


def test_sample_alignment(capture: MultiRadiodCapture, frequency_hz: float):
    """
    Test Stage 1: Sample alignment between two radiod instances.

    Args:
        capture: MultiRadiodCapture instance
        frequency_hz: Frequency to tune for alignment test
    """
    from phase_engine.calibration import SampleAligner

    logger.info("=" * 60)
    logger.info("STAGE 1: Sample Alignment Test")
    logger.info("=" * 60)

    # Tune to test frequency
    logger.info(f"Tuning to {frequency_hz/1e6:.3f} MHz...")
    channels = capture.tune_all(
        frequency_hz=frequency_hz,
        preset="am",
        sample_rate=12000,
    )

    # Wait for channels to stabilize
    time.sleep(1.0)

    # Capture samples
    samples = capture.capture_samples(channels, duration_s=2.0)

    # Clean up channels
    capture.remove_channels(channels)

    # Check we got samples from both
    names = list(samples.keys())
    if len(names) < 2:
        logger.error("Need samples from at least 2 radiod instances")
        return None

    for name, samps in samples.items():
        logger.info(f"  {name}: {len(samps)} samples, " f"power={np.mean(np.abs(samps)**2):.2e}")

    # Run sample alignment
    aligner = SampleAligner(sample_rate=12000)

    reference = names[0]
    target = names[1]

    logger.info(f"\nAligning {target} relative to {reference}...")

    result = aligner.calibrate_pair(
        reference=samples[reference],
        target=samples[target],
        reference_name=reference,
        target_name=target,
        frequency_hz=frequency_hz,
    )

    logger.info(f"\nAlignment Result:")
    logger.info(
        f"  Delay: {result.delay_samples} samples " f"({result.delay_samples / 12000 * 1e6:.1f} µs)"
    )
    logger.info(f"  Phase: {result.phase_offset_deg:.2f}°")
    logger.info(f"  Correlation peak: {result.correlation_peak:.4f}")
    logger.info(f"  Confidence: {result.confidence:.2f}")

    return result


def test_terrestrial_calibration(
    capture: MultiRadiodCapture,
    qth_lat: float,
    qth_lon: float,
    antenna_positions: Dict[str, tuple],
    reference_antenna: str,
):
    """
    Test Stage 2: Full terrestrial calibration.

    Args:
        capture: MultiRadiodCapture instance
        qth_lat: Receiver latitude
        qth_lon: Receiver longitude
        antenna_positions: Dict of antenna name -> (x, y, z) in meters
        reference_antenna: Name of reference antenna
    """
    from phase_engine.calibration import run_terrestrial_calibration

    logger.info("=" * 60)
    logger.info("STAGE 2: Terrestrial Calibration Test")
    logger.info("=" * 60)

    def tune_and_capture(frequency_hz: float, duration_s: float) -> Dict[str, np.ndarray]:
        """Callback for calibration routine."""
        channels = capture.tune_all(
            frequency_hz=frequency_hz,
            preset="am",
            sample_rate=12000,
        )
        time.sleep(0.5)  # Let channels stabilize

        samples = capture.capture_samples(channels, duration_s=duration_s)
        capture.remove_channels(channels)

        return samples

    result = run_terrestrial_calibration(
        qth_latitude=qth_lat,
        qth_longitude=qth_lon,
        antenna_positions=antenna_positions,
        reference_antenna=reference_antenna,
        tune_and_capture_fn=tune_and_capture,
        sample_rate=12000,
        capture_duration_s=2.0,
        max_distance_km=100.0,  # Wider search for testing
        min_power_kw=1.0,  # Lower threshold for testing
    )

    logger.info(f"\nCalibration Result:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Message: {result.message}")
    logger.info(f"  RMS Error: {result.rms_error_deg:.2f}°")

    if result.sources:
        logger.info(f"\nCalibration Sources Used:")
        for s in result.sources.stations:
            logger.info(
                f"  {s.call_sign} {s.frequency_khz} kHz @ {s.azimuth_deg:.0f}° "
                f"({s.distance_km:.0f} km)"
            )

    logger.info(f"\nPer-Antenna Results:")
    for name, cal in result.results.items():
        logger.info(f"  {name}:")
        logger.info(f"    Delay: {cal.delay_samples} samples")
        logger.info(f"    Phase offset: {cal.phase_offset_deg:.2f}°")
        logger.info(f"    Residual error: {cal.residual_error_deg:.2f}°")
        logger.info(f"    Confidence: {cal.confidence:.2f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test phase-engine calibration with multiple radiod instances"
    )
    parser.add_argument(
        "--radiod1", default="bee1-status.local", help="First radiod status address"
    )
    parser.add_argument(
        "--radiod2", default="bee4-status.local", help="Second radiod status address"
    )
    parser.add_argument(
        "--test-freq",
        type=float,
        default=None,
        help="Test frequency in MHz for sample alignment (default: auto-discover AM)",
    )
    parser.add_argument(
        "--qth-lat", type=float, default=39.0997, help="Receiver latitude (default: Kansas City)"
    )
    parser.add_argument(
        "--qth-lon", type=float, default=-94.5786, help="Receiver longitude (default: Kansas City)"
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "both"],
        default="both",
        help="Which calibration stage to test",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Define radiod instances
    # Names correspond to antenna positions
    instances = [
        RadiodInstance(name="bee1", status_address=args.radiod1),
        RadiodInstance(name="bee4", status_address=args.radiod2),
    ]

    # Antenna positions (placeholder - adjust for your actual array)
    # X = East, Y = North, Z = Up (meters)
    antenna_positions = {
        "bee1": (0.0, 0.0, 0.0),  # Reference
        "bee4": (10.0, 0.0, 0.0),  # 10m East of reference
    }
    reference_antenna = "bee1"

    # Create capture manager
    capture = MultiRadiodCapture(instances)

    try:
        # Connect to all radiod instances
        if not capture.connect():
            logger.error("Failed to connect to all radiod instances")
            return 1

        if args.stage in ("1", "both"):
            # Stage 1: Sample alignment
            if args.test_freq:
                test_freq = args.test_freq * 1e6
            else:
                # Use a common AM frequency for testing
                # TODO: Auto-discover from FCC database
                test_freq = 810e3  # 810 kHz - common AM frequency

            result = test_sample_alignment(capture, test_freq)
            if result is None:
                logger.error("Sample alignment test failed")
                return 1

        if args.stage in ("2", "both"):
            # Stage 2: Full terrestrial calibration
            result = test_terrestrial_calibration(
                capture=capture,
                qth_lat=args.qth_lat,
                qth_lon=args.qth_lon,
                antenna_positions=antenna_positions,
                reference_antenna=reference_antenna,
            )

            if not result.success:
                logger.warning("Terrestrial calibration did not fully succeed")
                # Continue anyway for debugging

        logger.info("\n" + "=" * 60)
        logger.info("Calibration test complete!")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return 1

    finally:
        capture.disconnect()


if __name__ == "__main__":
    sys.exit(main())
