# Phase Engine

**Coherent HF Phased Array Middleware**

Transform multiple GPSDO-locked RX888 SDRs into a single, coherent virtual receiver for diversity reception, beamforming, and interferometry.

## Status

**Early Development** - Architecture design phase.

## Overview

The Phase Engine acts as a "Virtual Radio Proxy" between physical `ka9q-radio` instances and scientific applications like `hf-timestd`. The application sees a single radio while the middleware manages synchronization and coherent combination of multiple physical streams.

```text
┌─────────────┐                              ┌──────────────────┐
│  hf-timestd │ <== (Fused RTP Stream) <==== │   phase-engine   │
│             │ ==> (Single Tune Cmd)  ====> │                  │
└─────────────┘                              └────────┬─────────┘
                                                      │
                              ┌────────────────┬──────┴──────┬────────────────┐
                              ▼                ▼             ▼                ▼
                         ┌─────────┐      ┌─────────┐   ┌─────────┐      ┌─────────┐
                         │ radiod  │      │ radiod  │   │ radiod  │      │ radiod  │
                         │ (RX888) │      │ (RX888) │   │ (RX888) │      │ (RX888) │
                         │  North  │      │  South  │   │  East   │      │  West   │
                         └─────────┘      └─────────┘   └─────────┘      └─────────┘
                              │                │             │                │
                              └────────────────┴──────┬──────┴────────────────┘
                                                      │
                                                 ┌────┴────┐
                                                 │  GPSDO  │
                                                 │ (Shared)│
                                                 └─────────┘
```

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

On startup, a calibration routine resolves the "Integer Sample Ambiguity":

1. Tune all radios to a strong wideband signal
2. FFT-based cross-correlation on 1 second of I/Q data
3. Output: Integer delay (Δn samples) and phase offset (Δφ degrees)
4. Feed constants into DSP block

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
│       ├── __init__.py
│       ├── control/           # Control plane (command proxy)
│       │   ├── __init__.py
│       │   ├── proxy.py       # Virtual radio proxy
│       │   └── radio_registry.py
│       ├── data/              # Data plane (DSP core)
│       │   ├── __init__.py
│       │   ├── ring_buffer.py
│       │   ├── combiner.py    # Coherent combination
│       │   └── rtp_handler.py
│       ├── calibration/       # Startup calibration
│       │   ├── __init__.py
│       │   └── sample_align.py
│       └── cli.py             # Command line interface
├── tests/
├── config/
│   └── phase-engine.toml.template
├── systemd/
│   └── phase-engine.service
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
