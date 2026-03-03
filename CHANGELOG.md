# Changelog

## [1.3.0] - 2026-03-03

### Summary

Major overhaul of the control plane and channel lifecycle to achieve full
transparent interoperability with `ka9q-python` and `hf-timestd`.  Phase Engine
now behaves as a drop-in replacement for a physical `radiod` instance — clients
using standard `ka9q-python` `ensure_channel()` / `discover_channels()` work
without modification.

### Added

- **`sdnotify` module** — Pure-Python `sd_notify()` for systemd `READY=1` and
  `WATCHDOG=1` integration; no external dependency required.
- **`CalibrationResult` dataclass** (`dsp/calibration.py`) — Placeholder for
  structured calibration output.
- **Persistent channel state** — Active channels are saved to
  `/var/lib/phase-engine/channels.json` on every configuration change and
  automatically restored on daemon restart via `VirtualChannelManager.restore_channels()`.
- **Flywheel zero-fill** in `EgressLoop` — When a backend radiod source drops,
  the egress loop injects silence (zero-filled IQ) to keep the client's RTP
  session alive until the source recovers.
- **Health monitor thread** in `RadiodSource` — Detects stale RTP streams
  (no packets for 3 s) and automatically reconnects the `RadiodControl` socket
  and recreates physical channels on the backend.
- **`consume_samples()`** on `RadiodSource` — Destructive read that returns
  `(samples, rtp_timestamp)` for the egress loop, preventing stale data
  reprocessing.

### Fixed

- **TLV type mismatch** — `control/tlv.py` now imports `StatusType` from the
  canonical `ka9q.types` module instead of maintaining a divergent local copy.
  This fixes silent misparsing of `PRESET` (was 68, should be 85),
  `OUTPUT_ENCODING` (was 85, should be 107), and `GAIN` (was 68).
- **Preset null terminator** — Status broadcasts no longer append `\x00` to the
  preset string.  The previous behaviour caused `ensure_channel()` verification
  to fail (`'iq\x00' != 'iq'`).
- **`OUTPUT_ENCODING` in status broadcasts** — Added to `_broadcast_status_now()`
  so that `ensure_channel()` encoding verification passes.
- **Discovery poll response** — `_handle_command()` now recognises the
  `SSRC=0xFFFFFFFF` broadcast probe and immediately responds with current
  channel status, enabling `discover_channels()` to find phase-engine channels.
- **Non-blocking channel provisioning** — `_evaluate_channel()` assigns the
  output multicast address immediately (so status broadcasts advertise the
  channel right away) then provisions radiod sources in a background thread.
  Previously, the synchronous `open_channel()` call blocked the command listener
  thread for several seconds per channel, starving concurrent CMDs and discovery
  polls.
- **`get_combined_samples()` return type** — Now returns
  `Optional[Tuple[np.ndarray, int]]` (samples + RTP timestamp) instead of a bare
  `np.ndarray`, fixing a `ValueError: too many values to unpack` in the egress
  loop.
- **Shebang placement** in `cli.py` — Moved `#!/usr/bin/env python3` before
  `os.environ` thread-limit overrides so the shebang is effective.
- **Corrupted import** in `dsp/calibration.py` — Fixed garbled first line.

### Changed

- **Version bump** to 1.3.0 (`pyproject.toml`, `__init__.py`).
- **TLV type sets** in `control/tlv.py` promoted to module-level `frozenset`s
  (were recreated inside the decode loop on every field).
- **Per-packet logging** in `control/server.py` downgraded from `INFO` to
  `DEBUG` to reduce log noise in production.
- **Dead AM calibration code** removed from `engine.py`.

### Companion changes (other repositories)

These changes are required in lockstep for correct operation:

- **`ka9q-python`** (`discovery.py`): Bind the status listener socket to the
  multicast group address instead of `0.0.0.0`.  This prevents
  `discover_channels()` from seeing status packets leaked from unrelated radiod
  instances sharing the same port.
- **`hf-timestd`**: Reverted the temporary `engine_type='phase-engine'`
  workaround in `StreamRecorderV2` / `CoreRecorderV2`.  With the above fixes,
  `hf-timestd` uses standard `ensure_channel()` for both radiod and
  phase-engine without any special-casing.
