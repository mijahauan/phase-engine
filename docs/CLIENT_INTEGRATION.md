# Phase-Engine Client Integration Guide

This document describes the minimal changes required to integrate existing applications with phase-engine.

## Quick Start

**Zero-code integration**: Point your app's radiod address at phase-engine instead of radiod. Phase-engine proxies all requests and applies default spatial filtering.

```bash
# Before: app talks to radiod
RADIOD_HOSTNAME=bee1-hf-status.local

# After: app talks to phase-engine
RADIOD_HOSTNAME=phase-engine-status.local
```

---

## hf-timestd Integration

### Minimal Change (Config Only)

Edit `config/timestd-config.toml`:

```toml
[ka9q]
# Point to phase-engine instead of radiod
status_address = "phase-engine-status.local"

# Already exists - tells hf-timestd to expect 17 channels (one per broadcast)
source = "phase-engine"
```

**That's it.** Phase-engine will:
1. Accept channel requests from hf-timestd
2. Auto-compute beam azimuth from frequency → station mapping
3. Apply FOCUS mode toward the target station
4. Return combined IQ stream

### Enhanced Integration (Optional)

To pass explicit reception objectives, modify `src/hf_timestd/core/stream_recorder_v2.py`:

```python
# In _create_channel() method, around line 368

# Check if we have phase-engine capabilities
caps = self._control.get_capabilities() if hasattr(self._control, 'get_capabilities') else {}

kwargs = {
    "frequency_hz": float(self.config.frequency_hz),
    "preset": self.config.preset,
    "sample_rate": self.config.sample_rate,
    "agc_enable": self.config.agc_enable,
    "gain": self.config.gain,
    "destination": self.config.destination,
    "encoding": self.config.encoding,
    "timeout": 10.0,
    "frequency_tolerance": 1.0
}

# Add phase-engine extensions if available
if caps.get("backend") == "phase-engine":
    # BroadcastRegistry already computed beam_azimuth_deg
    if hasattr(self.config, 'beam_azimuth_deg') and self.config.beam_azimuth_deg is not None:
        kwargs["reception_mode"] = "focus"
        kwargs["target"] = self.config.beam_azimuth_deg
    
    # For shared frequencies, use adaptive nulling
    if hasattr(self.config, 'requires_discrimination') and self.config.requires_discrimination:
        kwargs["reception_mode"] = "adaptive"

self.channel_info = self._control.ensure_channel(**kwargs)
```

### Files Changed

| File | Change | Lines |
|------|--------|-------|
| `config/timestd-config.toml` | Update `status_address` | 1 |
| `core/stream_recorder_v2.py` | Optional: pass reception objectives | ~15 |

---

## SWL-ka9q Integration

### Minimal Change (Config Only)

Edit `.radiod-hostname` or set environment variable:

```bash
# Option 1: Edit config file
echo "phase-engine-status.local" > .radiod-hostname

# Option 2: Environment variable
export RADIOD_HOSTNAME=phase-engine-status.local
```

**That's it.** SWL-ka9q will work unchanged, with phase-engine applying default spatial filtering.

### Enhanced Integration (UI Controls)

To let users select beamforming options, modify `radiod_client.py`:

```python
# Add new parameters to get_or_create_channel()

def get_or_create_channel(radiod_host: str, frequency: float,
                          interface: Optional[str] = None,
                          rtp_destination: Optional[str] = None,
                          rtp_port: Optional[int] = None,
                          preset: str = DEFAULT_PRESET,
                          sample_rate: int = DEFAULT_SAMPLE_RATE,
                          gain: float = 30.0,
                          agc_enable: bool = False,
                          include_metrics: bool = False,
                          # NEW: Phase-engine extensions
                          reception_mode: Optional[str] = None,
                          target: Optional[str] = None,
                          null_targets: Optional[List] = None) -> Dict:
    """
    Get or create an audio channel via ka9q-python.
    
    Phase-engine extensions (ignored if connected to plain radiod):
        reception_mode: "omni", "focus", "null", "adaptive", or "auto"
        target: Station name, azimuth (degrees), or (lat, lon) tuple
        null_targets: List of interference sources to null
    """
    try:
        with RadiodControl(radiod_host) as control:
            kwargs = {
                "frequency_hz": frequency,
                "preset": preset,
                "sample_rate": sample_rate,
                "agc_enable": 1 if agc_enable else 0,
                "gain": gain,
                "ssrc": None
            }
            
            # Add phase-engine extensions if provided
            if reception_mode:
                kwargs["reception_mode"] = reception_mode
            if target:
                kwargs["target"] = target
            if null_targets:
                kwargs["null_targets"] = null_targets
            
            ssrc = control.create_channel(**kwargs)
        
        # ... rest of function unchanged
```

Then in `server.js`, add API endpoint for beamforming control:

```javascript
// Add to existing /api/tune endpoint or create new one
app.post('/api/tune-enhanced', express.json(), async (req, res) => {
    const { frequency, preset, receptionMode, target, nullTargets } = req.body;
    
    // Build Python command with optional phase-engine params
    let cmd = `${PYTHON_CMD} radiod_client.py --radiod-host ${RADIOD_HOSTNAME} ` +
              `get-or-create --frequency ${frequency} --preset ${preset || 'am'}`;
    
    if (receptionMode) {
        cmd += ` --reception-mode ${receptionMode}`;
    }
    if (target) {
        cmd += ` --target "${target}"`;
    }
    if (nullTargets && nullTargets.length > 0) {
        cmd += ` --null-targets "${nullTargets.join(',')}"`;
    }
    
    // Execute and return result
    // ...
});
```

### UI Enhancement (Optional)

Add to the web interface:

```html
<!-- Reception mode selector -->
<div class="beamforming-controls">
    <label>Reception Mode:</label>
    <select id="reception-mode">
        <option value="auto">Auto (default)</option>
        <option value="omni">Omni (no beamforming)</option>
        <option value="focus">Focus (beam toward station)</option>
        <option value="adaptive">Adaptive (auto-null interference)</option>
    </select>
    
    <label>Target Station:</label>
    <input type="text" id="target-station" placeholder="e.g., Radio Habana Cuba">
</div>
```

### Files Changed

| File | Change | Lines |
|------|--------|-------|
| `.radiod-hostname` | Update hostname | 1 |
| `radiod_client.py` | Add optional parameters | ~20 |
| `server.js` | Add enhanced tune endpoint | ~30 |
| `public/index.html` | Optional: UI controls | ~20 |

---

## Capability Detection Pattern

Both apps should use this pattern to gracefully handle phase-engine vs plain radiod:

```python
def get_backend_capabilities(control):
    """
    Detect whether we're connected to phase-engine or plain radiod.
    Returns capability dict with safe defaults for plain radiod.
    """
    try:
        if hasattr(control, 'get_capabilities'):
            return control.get_capabilities()
    except Exception:
        pass
    
    # Default: assume plain radiod
    return {
        "backend": "radiod",
        "n_antennas": 1,
        "dof": 0,
        "modes": ["omni"],
        "max_nulls": 0,
        "can_focus_and_null": False
    }


def create_channel_smart(control, frequency_hz, preset, sample_rate, 
                         target=None, null_targets=None):
    """
    Create channel with automatic capability detection.
    Uses phase-engine features if available, falls back gracefully.
    """
    caps = get_backend_capabilities(control)
    
    kwargs = {
        "frequency_hz": frequency_hz,
        "preset": preset,
        "sample_rate": sample_rate
    }
    
    if caps.get("backend") == "phase-engine":
        # Phase-engine available - use spatial filtering
        if target and "focus" in caps.get("modes", []):
            kwargs["reception_mode"] = "focus"
            kwargs["target"] = target
        
        if null_targets and caps.get("max_nulls", 0) > 0:
            kwargs["null_targets"] = null_targets[:caps["max_nulls"]]
            if "focus_null" in caps.get("modes", []):
                kwargs["reception_mode"] = "focus_null"
    
    return control.create_channel(**kwargs)
```

---

## Testing Integration

### Verify Phase-Engine Connection

```python
from phase_engine import PhaseEngineControl

control = PhaseEngineControl("phase-engine-status.local")
caps = control.get_capabilities()

print(f"Backend: {caps['backend']}")
print(f"Antennas: {caps['n_antennas']}")
print(f"DoF: {caps['dof']}")
print(f"Modes: {caps['modes']}")
```

### Test Channel Creation

```python
# Standard (works with radiod or phase-engine)
ch = control.create_channel(frequency_hz=10000000, preset="iq", sample_rate=24000)
print(f"SSRC: {ch.ssrc}, Address: {ch.multicast_address}")

# Enhanced (phase-engine only, ignored by radiod)
ch = control.create_channel(
    frequency_hz=10000000,
    preset="iq", 
    sample_rate=24000,
    reception_mode="focus",
    target="WWV"
)
print(f"Beam azimuth: {ch.beam_azimuth_deg}°")
```

---

## Summary

| App | Zero-Code Change | Enhanced Integration |
|-----|------------------|---------------------|
| **hf-timestd** | Change `status_address` in config | Pass `reception_mode` per channel |
| **SWL-ka9q** | Change `.radiod-hostname` | Add UI controls for beamforming |
| **Any ka9q app** | Point at phase-engine | Use extended `create_channel()` params |

Phase-engine is designed for **progressive enhancement**: apps work unchanged, and can opt into spatial filtering features as desired.
