# Phase-Engine Client API Specification

**Version:** 1.0.0-draft  
**Date:** 2026-01-23

## Overview

Phase-engine is a coherent phased array processor that sits between client applications and multiple radiod instances. It provides spatial filtering (beamforming, nulling) and diversity combining to improve reception.

This document specifies the client-facing API that applications use to request channels from phase-engine. The API is designed to be:

1. **Backwards compatible** with radiod — apps can use phase-engine as a drop-in replacement
2. **Progressively enhanced** — apps can opt into spatial filtering features
3. **Capability-aware** — apps can query what phase-engine can do

## Architecture

```
┌─────────────────┐
│  Client Apps    │  hf-timestd, SWL-ka9q, etc.
│  (unchanged)    │
└────────┬────────┘
         │ Standard ka9q-python API
         │ + optional extensions
         ▼
┌─────────────────┐
│  phase-engine   │  Coherent array processor
│  Control Proxy  │
└────────┬────────┘
         │ Internal: coordinates N radiod instances
         ▼
┌─────────────────┐
│  radiod × N     │  RX888 SDRs with GPSDO sync
└─────────────────┘
```

## Capability Model

Phase-engine capabilities depend on the number of antennas (radiod instances):

| Antennas | DoF | Capabilities |
|----------|-----|--------------|
| 2 | 1 | Beam OR null (not both), diversity combining, 1D AoA |
| 3 | 2 | Beam AND null, 2D AoA with ambiguity, MUSIC (2 sources) |
| 4 | 3 | Beam + 2 nulls, robust 2D AoA, MUSIC (3 sources), MVDR |
| N | N-1 | Up to N-1 nulls, resolve N-1 sources |

### Degrees of Freedom (DoF)

- **DoF = N_antennas - 1**
- Each null constraint consumes 1 DoF
- Focus (beam steering) uses array geometry but doesn't consume DoF
- Adaptive algorithms (MVDR) use DoF to auto-null interferers

---

## API Reference

### 1. Capability Query

Query phase-engine capabilities before requesting channels.

```python
from phase_engine import PhaseEngineControl

control = PhaseEngineControl("phase-engine-status.local")
caps = control.get_capabilities()
```

**Response:**

```python
{
    "backend": "phase-engine",      # "phase-engine" or "radiod" (passthrough)
    "version": "1.0.0",
    "n_antennas": 4,
    "dof": 3,                       # degrees of freedom
    
    # Available reception modes
    "modes": ["omni", "focus", "null", "adaptive"],
    
    # Spatial filtering limits
    "max_simultaneous_beams": 1,    # per frequency
    "max_nulls": 3,                 # per channel (= dof)
    "can_focus_and_null": true,     # dof >= 2
    
    # Direction finding
    "can_aoa_estimate": true,
    "max_resolvable_sources": 3,    # = dof
    "aoa_algorithms": ["music", "esprit", "beamscan"],
    
    # Combining methods
    "combining_methods": ["mrc", "egc", "selection"],
    
    # Array center location (for client-side azimuth calculation to targets)
    "array_center_lat": 38.918461,
    "array_center_lon": -92.127974,
    
    # Calibration status (steering vectors derived from known sources)
    "calibration": {
        "status": "calibrated",         # "uncalibrated", "calibrating", "calibrated"
        "last_calibration": "2026-01-23T00:30:00Z",
        "calibration_sources": [        # Groundwave AM stations used for phase cal
            {"name": "KFRU", "frequency_khz": 1400, "azimuth_deg": 15.2},
            {"name": "KLIK", "frequency_khz": 1240, "azimuth_deg": 95.7},
            {"name": "KWOS", "frequency_khz": 950, "azimuth_deg": 178.3}
        ],
        "calibrated_azimuths_deg": [15.2, 95.7, 178.3],  # Directions with measured steering vectors
        "interpolation_range_deg": 180   # Can interpolate within this span
    }
}
```

If connected to plain radiod (no phase-engine), returns:

```python
{
    "backend": "radiod",
    "version": "radiod-version",
    "n_antennas": 1,
    "dof": 0,
    "modes": ["omni"],
    "max_nulls": 0,
    "can_focus_and_null": false,
    "can_aoa_estimate": false
}
```

---

### 2. Channel Creation (Standard)

Standard channel creation works identically to radiod:

```python
channel = control.create_channel(
    frequency_hz=9650000,
    preset="am",
    sample_rate=12000,
    agc_enable=1,
    gain=30.0
)
```

Phase-engine uses **default reception mode** (typically "omni" or auto-focus based on known station azimuths).

---

### 3. Channel Creation (Extended)

Apps can request specific spatial filtering:

```python
channel = control.create_channel(
    frequency_hz=9650000,
    preset="am",
    sample_rate=12000,
    
    # Phase-engine extensions (ignored by plain radiod)
    reception_mode="focus",         # See Reception Modes below
    target="WWV",                   # Station name, azimuth, or coordinates
    null_targets=["BPM", 120.0],    # Optional: stations/azimuths to null
    combining_method="mrc",         # mrc | egc | selection
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reception_mode` | str | "auto" | Reception mode (see below) |
| `target` | str/float/tuple | None | Beam target: station name, azimuth (deg), or (lat, lon) |
| `null_targets` | list | [] | List of interference sources to null |
| `combining_method` | str | "mrc" | How to combine antenna signals |
| `priority` | int | 0 | Resource allocation priority (higher = more important) |

---

### 4. Reception Modes

| Mode | Description | DoF Required |
|------|-------------|--------------|
| `"omni"` | No spatial filtering, diversity combining only | 0 |
| `"focus"` | Steer beam toward target | 0 (uses geometry) |
| `"null"` | Place null toward interferer(s) | 1 per null |
| `"focus_null"` | Focus on target AND null interferer(s) | 1 per null |
| `"adaptive"` | MVDR: auto-null interferers while preserving target | 1+ |
| `"auto"` | Phase-engine chooses based on frequency/station | 0 |

**Mode Selection Logic for "auto":**

1. If frequency maps to known station → `"focus"` toward that station
2. If multiple stations on frequency → `"adaptive"` (MVDR)
3. Otherwise → `"omni"`

---

### 5. Target Specification

The `target` parameter accepts multiple formats:

```python
# By station name (phase-engine looks up azimuth)
target="WWV"
target="Radio Habana Cuba"

# By azimuth (degrees from north, clockwise)
target=45.0

# By coordinates (lat, lon)
target=(40.67805, -105.04719)

# By broadcast ID (for hf-timestd)
target="WWV_10000"
```

---

### 6. Null Specification

The `null_targets` parameter accepts a list:

```python
# Mixed formats allowed
null_targets=[
    "BPM",          # Station name
    120.0,          # Azimuth in degrees
    (34.95, 109.54) # Coordinates
]
```

**Constraints:**
- Maximum nulls = DoF (e.g., 3 nulls with 4 antennas)
- If `reception_mode="focus_null"`, one DoF is implicitly used for focus
- Excess null requests are silently dropped (lowest priority first)

---

### 7. Channel Info Response

Extended channel info includes spatial filtering status:

```python
{
    "ssrc": 0x12345678,
    "frequency_hz": 9650000,
    "multicast_address": "239.1.2.100",
    "port": 5004,
    "sample_rate": 12000,
    "preset": "am",
    
    # Phase-engine extensions
    "reception_mode": "focus_null",
    "beam_azimuth_deg": 265.3,
    "beam_target": "WWV",
    "null_azimuths_deg": [85.2],
    "null_targets": ["BPM"],
    "combining_method": "mrc",
    "estimated_gain_db": 6.0,       # vs single antenna
    "estimated_null_depth_db": 25.0 # interference suppression
}
```

---

### 8. Dynamic Reconfiguration

Change spatial filtering on an active channel:

```python
control.reconfigure_channel(
    ssrc=0x12345678,
    reception_mode="adaptive",
    target="WWVH",
    null_targets=[]
)
```

Useful for SWL-ka9q "try different modes" UI.

---

### 9. AoA Estimation (Optional)

Request direction-of-arrival estimation:

```python
aoa = control.estimate_aoa(
    frequency_hz=9650000,
    duration_sec=5.0,
    algorithm="music",      # music | esprit | beamscan
    max_sources=3
)
```

**Response:**

```python
{
    "frequency_hz": 9650000,
    "timestamp": "2026-01-23T01:15:00Z",
    "sources": [
        {"azimuth_deg": 265.3, "power_db": -45.2, "confidence": 0.95},
        {"azimuth_deg": 85.1, "power_db": -52.8, "confidence": 0.78},
    ],
    "noise_floor_db": -80.0,
    "algorithm": "music"
}
```

---

## Integration Examples

### hf-timestd (Minimal Change)

```python
# config/timestd-config.toml
[ka9q]
status_address = "phase-engine-status.local"  # Point to phase-engine
source = "phase-engine"

# In code: BroadcastRegistry already computes beam_azimuth_deg per broadcast
# Phase-engine uses this automatically when source="phase-engine"
```

### SWL-ka9q (Optional Enhancement)

```python
# radiod_client.py - add optional reception_mode parameter
def get_or_create_channel(radiod_host, frequency, ..., reception_mode=None, target=None):
    kwargs = {
        "frequency_hz": frequency,
        "preset": preset,
        "sample_rate": sample_rate,
    }
    
    # Add phase-engine extensions if provided
    if reception_mode:
        kwargs["reception_mode"] = reception_mode
    if target:
        kwargs["target"] = target
    
    with RadiodControl(radiod_host) as control:
        ssrc = control.create_channel(**kwargs)
    ...
```

UI could add:
- "Beamform" toggle → `reception_mode="focus"`
- "Null interference" toggle → `reception_mode="adaptive"`
- Station selector → `target=station_name`

---

## Error Handling

Phase-engine returns standard errors plus spatial-specific ones:

| Error | Description |
|-------|-------------|
| `INSUFFICIENT_DOF` | Requested more nulls than available DoF |
| `UNKNOWN_TARGET` | Station name not in database |
| `INVALID_AZIMUTH` | Azimuth outside 0-360 range |
| `CALIBRATION_REQUIRED` | Array not calibrated for this frequency |
| `MODE_NOT_SUPPORTED` | Reception mode not available |

---

## Backwards Compatibility

1. **Plain radiod clients** work unchanged — phase-engine proxies requests
2. **Extended parameters** are ignored by plain radiod
3. **Capability query** returns `backend="radiod"` if no phase-engine
4. **Apps should check capabilities** before using extended features

```python
caps = control.get_capabilities()
if caps.get("backend") == "phase-engine" and caps.get("dof", 0) >= 1:
    # Can use spatial filtering
    channel = control.create_channel(..., reception_mode="focus")
else:
    # Fallback to standard
    channel = control.create_channel(...)
```

---

## Future Extensions

- **Multi-beam**: Simultaneous beams to different targets (requires more antennas)
- **Wideband nulling**: Null across frequency range
- **Adaptive tracking**: Auto-adjust beam/null as propagation changes
- **Distributed arrays**: Coordinate multiple phase-engine instances

---

## Appendix: Station Database

Phase-engine maintains a database of known stations with coordinates for automatic azimuth calculation. Initial set includes:

| Station | Location | Coordinates |
|---------|----------|-------------|
| WWV | Fort Collins, CO | 40.678, -105.047 |
| WWVH | Kekaha, HI | 21.988, -159.762 |
| CHU | Ottawa, ON | 45.295, -75.754 |
| BPM | Pucheng, China | 34.948, 109.542 |

Apps can register additional stations or provide coordinates directly.
