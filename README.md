# Phase Engine

**Coherent HF Phased Array Middleware**

Transform multiple GPSDO-locked RX888 SDRs into a single, coherent virtual receiver for diversity reception, beamforming, and interferometry.

## Status

**Active Development** - Core functionality implemented and tested.

## Overview

Phase Engine acts as a drop-in replacement for `ka9q-radio` that coordinates multiple physical SDRs into a coherent phased array. Client applications use the same API as `ka9q-python` with optional extensions for spatial filtering.

```text
┌─────────────────┐
│  Client Apps    │  hf-timestd, SWL-ka9q, etc.
│  (unchanged)    │
└────────┬────────┘
         │ PhaseEngineControl / PhaseEngineStream
         │ (API-compatible with ka9q-python)
         ▼
┌─────────────────────────────────────────────────────────────┐
│                      phase-engine                            │
├─────────────────────────────────────────────────────────────┤
│  RadiodSource(A)    RadiodSource(B)    RadiodSource(C)      │
│       │                  │                  │                │
│       ▼                  ▼                  ▼                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Sample Alignment Layer                  │    │
│  │   (cross-correlate to find integer sample delays)   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Phase Combiner (×17)                    │    │
│  │   (per-broadcast beam steering and combining)        │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                 17 Combined Outputs                          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  radiod × N     │  RX888 SDRs with shared GPSDO             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### For Client Applications

Replace `RadiodControl` with `PhaseEngineControl`:

```python
# Before: direct radiod connection
from ka9q import RadiodControl
control = RadiodControl("bee1-status.local")

# After: phase-engine (drop-in replacement)
from phase_engine import PhaseEngineControl
control = PhaseEngineControl("phase-engine-status.local")

# Standard channel creation works unchanged
channel = control.create_channel(frequency_hz=10e6, preset="iq")

# Extended: spatial filtering (optional)
channel = control.create_channel(
    frequency_hz=10e6,
    preset="iq",
    reception_mode="focus",    # Beam toward target
    target="WWV",              # Station name, azimuth, or (lat, lon)
    null_targets=["BPM"],      # Interference to null
    combining_method="mrc",    # Maximum ratio combining
)
```

### Reception Modes

| Mode | Description | DoF Required |
|------|-------------|--------------|
| `omni` | No spatial filtering, diversity combining only | 0 |
| `focus` | Steer beam toward target | 0 (geometry) |
| `null` | Place null toward interferer(s) | 1 per null |
| `focus_null` | Focus on target AND null interferer(s) | 1 per null |
| `adaptive` | MVDR: auto-null interferers | 1+ |
| `auto` | Phase-engine chooses based on frequency | 0 |

## Architecture

### Control Plane (The Proxy)

- **Upstream Interface**: Listen on virtual control port (UDP)
- **Logic**: Accept standard radio commands (`tune`, `gain`, etc.)
- **Downstream Action**: Broadcast commands to all physical radios simultaneously

### Data Plane (The DSP Core)

- **Ingest**: Subscribe to RTP streams from all radios
- **Ring Buffers**: Circular buffers to align streams by timestamp
- **Mixer Math**: Coherent combination algorithms
- **Egress**: Repackage fused signal as valid RTP stream

## Combination Modes

| Mode | Formula | Use Case |
|------|---------|----------|
| **MRC** (Max Ratio Combining) | `Σ(wᵢ·sᵢ)` weighted by SNR | Maximize SNR, best for fading |
| **EGC** (Equal Gain Combining) | `Σ(sᵢ)/N` | Simple sum, identical antennas |
| **Selection** | `max(SNR)` | Use best antenna only |
| **Nulling** | `s₁ - e^(jφ)·s₂` | Cancel interference |

## Capabilities by Antenna Count

| Antennas | DoF | Capabilities |
|----------|-----|--------------|
| 2 | 1 | Beam OR null, diversity gain, 1D angle |
| 3 | 2 | Beam AND null, 2D angle (Az+El), MUSIC algorithm |
| 4 | 3 | Multiple nulls, robust 2D, redundancy |

## User Controls

- **Virtual Rotator**: 360° phase dial for manual beam steering
- **Null Button**: Instant interference cancellation
- **Diversity Mode**: MRC/EGC/Selection/Anti-Phase selector
- **Look Angle**: Elevation slider (0-90°) for take-off angle

## Automated Algorithms

- **Adaptive Null Steering**: MVDR/LMS for automatic RFI tracking
- **Beam Tracking**: Gradient ascent to follow ionospheric drift
- **Polarization Matching**: Compensate for Faraday rotation
- **ToA Estimation**: Multi-frequency interferometry for hop discrimination

## Calibration

Phase-engine uses **terrestrial calibration** with local AM broadcast stations to establish array geometry before operation. This approach requires no timing information from the upstream application.

### Terrestrial Calibration (Primary Method)

A **two-stage** calibration using ground-wave signals from nearby AM broadcast towers:

**Stage 1: Sample Alignment** (resolves ADC start-time ambiguity)
1. Auto-discover 2-3 local AM stations (≥5 kW, ≤50 km) from FCC database
2. Tune all radios to the strongest source
3. Cross-correlate streams to find **integer sample delays** between antennas
4. Apply delays to align streams in time

**Stage 2: Phase Calibration** (resolves cable/analog offsets)
1. Tune to each AM source (now with aligned streams)
2. Measure inter-antenna phase differences
3. Compare to predicted phases from geometry (tower bearing + antenna positions)
4. Solve for **residual phase offsets** per antenna

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    TERRESTRIAL CALIBRATION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   AM Tower 1 (KCMO 710 kHz)          AM Tower 2 (KMBZ 980 kHz)     │
│   @ 285° azimuth, 12 km              @ 45° azimuth, 8 km           │
│         ╲                                   ╱                       │
│          ╲  ground wave                    ╱  ground wave           │
│           ╲                               ╱                         │
│            ╲                             ╱                          │
│             ╲     ┌─────────────┐       ╱                           │
│              ╲    │   Antenna   │      ╱                            │
│               ╲   │    Array    │     ╱                             │
│                ╲  │  [N][S][E]  │    ╱                              │
│                 ╲ └─────────────┘   ╱                               │
│                  ╲                 ╱                                │
│                   ╲               ╱                                 │
│    Measured Δφ = Predicted Δφ + Calibration Offset                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Ground-wave arrival angles are stable and predictable (azimuth from geometry, elevation ≈ 0°)
- No dependency on HF propagation or upstream timing
- FCC database provides precise tower coordinates
- Works 24/7 (with clear-channel avoidance at night)

**Clear-Channel Protection:** At night, ~45 clear-channel frequencies are excluded to avoid skywave interference from distant 50 kW stations.

### Signal-Based Calibration (Alternative)

For initial sample alignment, FFT-based cross-correlation on a known wideband signal:

1. Tune all radios to a strong signal (e.g., 10 MHz WWV)
2. Cross-correlate 1 second of I/Q data
3. Output: Integer delay (Δn samples) and phase offset (Δφ degrees)

### Beam Steering After Calibration

Once calibrated, the array can steer toward any azimuth/elevation purely from geometry:

```python
# Steer toward WWV (Fort Collins, CO) from Kansas City
wwv_bearing = 280  # degrees (computed from lat/lon)
phase_engine.steer_to(azimuth=wwv_bearing, elevation=15)
```

The upstream application (`hf-timestd`) provides target station identity; `phase-engine` computes and applies the steering phases.

## Requirements

- Python 3.11+
- NumPy (DSP operations)
- `ka9q-python` (radio control library)
- GPSDO-locked RX888 SDRs sharing a common clock

## Project Structure

```
phase-engine/
├── src/
│   └── phase_engine/
│       ├── __init__.py           # Main exports
│       ├── engine.py             # PhaseEngine orchestrator
│       ├── client/               # Client API (ka9q-python compatible)
│       │   ├── __init__.py
│       │   ├── control.py        # PhaseEngineControl
│       │   └── stream.py         # PhaseEngineStream
│       ├── sources/              # Radiod source management
│       │   ├── __init__.py
│       │   ├── radiod_source.py  # Per-radiod stream manager
│       │   └── broadcasts.py     # Station/broadcast database
│       ├── combiner/             # Phase combining
│       │   ├── __init__.py
│       │   └── phase_combiner.py # Per-broadcast beam steering
│       ├── calibration/          # Array calibration
│       │   ├── __init__.py
│       │   ├── sample_align.py   # Cross-correlation alignment
│       │   ├── sources.py        # FCC AM station database
│       │   └── terrestrial.py    # Geometric calibration
│       ├── control/              # Legacy control plane
│       │   └── radio_registry.py
│       ├── data/                 # DSP utilities
│       │   ├── ring_buffer.py
│       │   └── combiner.py
│       └── cli.py                # Command line interface
├── scripts/
│   ├── test_engine.py            # Full engine test
│   └── test_client_api.py        # Client API test
├── docs/
│   ├── API_SPECIFICATION.md      # Client API reference
│   └── CLIENT_INTEGRATION.md     # Integration guide
├── config/
│   └── phase-engine.toml.template
├── pyproject.toml
├── README.md
└── LICENSE
```

## Related Projects

- [hf-timestd](https://github.com/mijahauan/hf-timestd) - HF time standard application
- [ka9q-radio](https://github.com/ka9q/ka9q-radio) - Software-defined radio framework
- [ka9q-python](https://github.com/mijahauan/ka9q-python) - Python bindings for ka9q-radio

## License

MIT
