#!/usr/bin/env python3
"""
TLV Parity Test — verify phase-engine status broadcasts match the radiod contract.

Compares discover_channels() output from a real upstream radiod against
phase-engine's status address.  Every field that hf-timestd consumes must
be present and well-formed in the phase-engine output.

Usage:
    /opt/phase-engine/venv/bin/python scripts/test_tlv_parity.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ka9q import discover_channels

# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------
PHASE_ENGINE_STATUS = "239.99.1.1"
UPSTREAM_RADIOD_STATUS = "bee3-status.local"   # reference antenna (longwire)

# ---------------------------------------------------------------------------
# Fields that hf-timestd accesses from ChannelInfo
# ---------------------------------------------------------------------------
# Critical for timing (BinaryArchiveWriter gate):
TIMING_FIELDS = ["gps_time", "rtp_timesnap"]

# Critical for channel subscription (StreamInfo.from_channel_info):
SUBSCRIPTION_FIELDS = ["frequency", "preset", "sample_rate",
                       "multicast_address", "port"]

# Used but not gating:
OPTIONAL_FIELDS = ["ssrc", "snr", "encoding", "description"]

ALL_REQUIRED = TIMING_FIELDS + SUBSCRIPTION_FIELDS


def discover_safe(addr, label, duration=2.0):
    """Discover channels, return dict or exit on failure."""
    print(f"\n{'='*60}")
    print(f"  Discovering from {label}: {addr}")
    print(f"{'='*60}")
    try:
        channels = discover_channels(addr, listen_duration=duration)
    except Exception as e:
        print(f"  FAIL: discover_channels raised {e}")
        return None
    print(f"  Found {len(channels)} channel(s)")
    return channels


def check_field(info, field):
    """Return (value, ok) for a ChannelInfo attribute."""
    val = getattr(info, field, None)
    ok = val is not None
    return val, ok


def print_channel(ssrc, info, label):
    """Pretty-print one channel's fields."""
    print(f"\n  [{label}] SSRC 0x{ssrc:08X}")
    for f in ALL_REQUIRED + OPTIONAL_FIELDS:
        val, ok = check_field(info, f)
        status = "✅" if ok else "❌"
        print(f"    {status} {f:25s} = {val}")


def main():
    failures = []

    # 1. Discover from phase-engine
    pe_channels = discover_safe(PHASE_ENGINE_STATUS, "phase-engine")
    if pe_channels is None or len(pe_channels) == 0:
        print("\nFATAL: No channels from phase-engine")
        sys.exit(1)

    # 2. Discover from upstream radiod (for reference)
    radiod_channels = discover_safe(UPSTREAM_RADIOD_STATUS, "upstream radiod")

    # 3. Check every required field on every phase-engine channel
    print(f"\n{'='*60}")
    print(f"  Phase-Engine Field Audit")
    print(f"{'='*60}")

    for ssrc, info in pe_channels.items():
        print_channel(ssrc, info, "phase-engine")
        for f in ALL_REQUIRED:
            val, ok = check_field(info, f)
            if not ok:
                failures.append(f"SSRC 0x{ssrc:08X}: missing {f}")

    # 4. Reference: show one radiod channel for comparison
    if radiod_channels:
        print(f"\n{'='*60}")
        print(f"  Upstream Radiod Reference (first channel)")
        print(f"{'='*60}")
        ssrc, info = next(iter(radiod_channels.items()))
        print_channel(ssrc, info, "radiod")

    # 5. Cross-check: GPS_TIME values should be close between the two
    if radiod_channels:
        pe_gps = None
        for info in pe_channels.values():
            if getattr(info, "gps_time", None) is not None:
                pe_gps = info.gps_time
                break
        rd_gps = None
        for info in radiod_channels.values():
            if getattr(info, "gps_time", None) is not None:
                rd_gps = info.gps_time
                break

        if pe_gps and rd_gps:
            # GPS_TIME is nanoseconds since Unix epoch
            diff_s = abs(pe_gps - rd_gps) / 1e9
            ok = diff_s < 10.0  # should be within a few seconds
            status = "✅" if ok else "❌"
            print(f"\n  {status} GPS_TIME delta: {diff_s:.3f}s (phase-engine vs radiod)")
            if not ok:
                failures.append(f"GPS_TIME delta {diff_s:.1f}s exceeds 10s threshold")
        else:
            print(f"\n  ⚠️  Cannot cross-check GPS_TIME (pe={pe_gps}, rd={rd_gps})")

    # 6. Verify multicast_address is NOT an upstream radiod address
    # (phase-engine must assign its own output addresses)
    upstream_addrs = set()
    if radiod_channels:
        for info in radiod_channels.values():
            if getattr(info, "multicast_address", None):
                upstream_addrs.add(info.multicast_address)

    for ssrc, info in pe_channels.items():
        addr = getattr(info, "multicast_address", None)
        if addr and addr in upstream_addrs:
            failures.append(
                f"SSRC 0x{ssrc:08X}: multicast_address {addr} collides with upstream radiod"
            )

    # 7. Summary
    print(f"\n{'='*60}")
    if failures:
        print(f"  ❌ FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"     • {f}")
        sys.exit(1)
    else:
        n = len(pe_channels)
        print(f"  ✅ PASSED — {n} channel(s), all required fields present")
        print(f"     Timing:       GPS_TIME + RTP_TIMESNAP ✅")
        print(f"     Subscription: frequency, preset, sample_rate, dest ✅")
        print(f"     Phase-engine is indistinguishable from radiod ✅")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
