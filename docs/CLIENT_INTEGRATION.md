# Phase-Engine Client Integration Guide

**Date:** 2026-03-01  
**Version:** 1.1.0

This document describes how `ka9q-python` clients interact with phase-engine, covering the actual wire protocol, channel lifecycle, and multicast address assignment.

## Quick Start

**Zero-code integration**: Point your app's radiod status address at phase-engine's multicast IP instead of a physical radiod. Phase-engine proxies all `ka9q-radio` TLV commands and applies spatial beamforming transparently.

```bash
# Before: app talks directly to radiod
RADIOD_STATUS_ADDR=bee1-status.local

# After: app talks to phase-engine
RADIOD_STATUS_ADDR=239.99.1.1   # phase-engine status multicast IP
```

> **Note:** `status_address` in phase-engine config must be a **multicast IP address** (e.g. `239.99.1.1`), not a hostname. `inet_aton` is used internally and rejects hostnames.

---

## How the Protocol Works

Understanding the actual wire protocol is essential for integration. This differs from a plain `radiod` connection in one important way: **clients never supply a destination address**.

### Channel Request Lifecycle

```
Client (hf-timestd / ka9q-python)          Phase Engine
─────────────────────────────────          ────────────
1. TLV CMD  ──────────────────────────────►  _handle_command()
   {SSRC, FREQ, PRESET, SAMPLE_RATE}          │
   (no destination)                           ├─ configure_channel(ssrc, params)
                                              │   └─ _evaluate_channel(ssrc)
                                              │       ├─ _assign_output_address(ssrc)
                                              │       │   (deterministic 239.x.x.x)
                                              │       └─ engine.open_channel(freq)
                                              │           (creates channel on all radiods)
2. TLV STATUS ACK  ◄─────────────────────────┤
   {SSRC, OUTPUT_DATA_DEST_SOCKET}            │  (includes assigned multicast addr)
                                              │
3. discover_channels() ──────────────────►   │
   (polls status multicast 239.99.1.1)        │
                                              ├─ _status_multicaster_loop()
4. ChannelInfo  ◄────────────────────────────┘  {SSRC, FREQ, DEST_SOCKET, SAMPRATE}

5. RadiodStream subscribes to
   assigned multicast address
   and receives combined RTP IQ
```

### Key Design Points

- **Clients never send `OUTPUT_DATA_DEST_SOCKET`** — phase-engine ignores it if present. The output address is always assigned by phase-engine.
- **Deterministic multicast addresses** — the same SSRC always maps to the same `239.x.x.x` group (SHA-256 of SSRC bytes). Restart-safe.
- **Lazy physical channel allocation** — `open_channel()` is only called when a client sends a frequency request. On startup, phase-engine calibrates and then sits idle.
- **SSRC `0xFFFFFFFF`** is a broadcast discovery probe from `ka9q` tools. Phase-engine silently ignores it.

---

## hf-timestd Integration

### Minimal Change (Config Only)

Edit `config/timestd-config.toml`:

```toml
[ka9q]
# Point to phase-engine status multicast IP (must be an IP, not a hostname)
status_address = "239.99.1.1"
```

**That's it.** `hf-timestd` via `ka9q-python` will:
1. Send `{SSRC, FREQ, PRESET, SAMPLE_RATE}` TLV CMD to port 5006
2. Receive STATUS ACK with the assigned `OUTPUT_DATA_DEST_SOCKET`
3. Alternatively call `discover_channels()` which polls the status multicast
4. Subscribe the `RadiodStream` to the assigned multicast address

Do **not** pass a `destination` kwarg to `create_channel()` — phase-engine does not accept it from clients.

### Enhanced Integration (Optional)

To pass explicit spatial processing hints, add `reception_mode` and `target` to the `create_channel()` call in `src/hf_timestd/core/stream_recorder_v2.py`:

```python
kwargs = {
    "frequency_hz": float(self.config.frequency_hz),
    "preset": self.config.preset,
    "sample_rate": self.config.sample_rate,
    "agc_enable": self.config.agc_enable,
    "gain": self.config.gain,
    # NOTE: do NOT pass destination= here
    "encoding": self.config.encoding,
    "timeout": 10.0,
}

# Add phase-engine spatial extensions if desired
if hasattr(self.config, 'beam_azimuth_deg') and self.config.beam_azimuth_deg is not None:
    kwargs["reception_mode"] = "focus"
    kwargs["target"] = self.config.beam_azimuth_deg

if hasattr(self.config, 'requires_discrimination') and self.config.requires_discrimination:
    kwargs["reception_mode"] = "adaptive"

self.channel_info = self._control.ensure_channel(**kwargs)
```

### Files Changed

| File | Change | Lines |
|------|--------|-------|
| `config/timestd-config.toml` | Update `status_address` to multicast IP | 1 |
| `core/stream_recorder_v2.py` | Optional: pass reception objectives | ~10 |

---

## SWL-ka9q Integration

### Minimal Change (Config Only)

Edit `.radiod-hostname` or set environment variable:

```bash
# Option 1: Edit config file
echo "239.99.1.1" > .radiod-hostname

# Option 2: Environment variable
export RADIOD_HOSTNAME=239.99.1.1
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

## TLV Wire Format Reference

Phase-engine speaks standard `ka9q-radio` TLV. The relevant type codes:

| Constant | Value | Description |
|---|---|---|
| `EOL` | 0 | End of TLV list |
| `COMMAND_TAG` | 1 | Echo tag for ACK matching |
| `OUTPUT_DATA_DEST_SOCKET` | 17 | AF(2) \| port(2) \| IPv4(4) — **assigned by phase-engine, not client** |
| `OUTPUT_SSRC` | 18 | Channel SSRC (client-allocated via `allocate_ssrc()`) |
| `OUTPUT_SAMPRATE` | 20 | Sample rate in Hz |
| `RADIO_FREQUENCY` | 33 | Center frequency as 64-bit double (Hz) |
| `DEMOD_TYPE` | 48 | Demodulator type (0 = IQ) |
| `PRESET` | 68 | Preset name string (NUL-terminated) |
| `OUTPUT_ENCODING` | 85 | Sample encoding (26993 = F32) |

Socket address encoding (`OUTPUT_DATA_DEST_SOCKET`, 8 bytes):
```
byte 0-1: AF_INET (big-endian uint16, = 2)
byte 2-3: port    (big-endian uint16)
byte 4-7: IPv4    (4 bytes, inet_aton format)
```

---

## Testing Integration

### Verify Phase-Engine is Running

```bash
# Check service status
sudo systemctl status phase-engine

# Tail logs - should show "idle, waiting for client requests" after calibration
sudo journalctl -u phase-engine -f
```

### Test Channel Creation

```python
from ka9q import RadiodControl

# Connect to phase-engine status multicast
control = RadiodControl("239.99.1.1")

# Standard ka9q-python channel creation — no destination kwarg
ssrc = control.allocate_ssrc(frequency_hz=10e6, preset="iq", sample_rate=24000)
channel_info = control.ensure_channel(ssrc)

print(f"SSRC: {ssrc}")
print(f"Assigned multicast: {channel_info.multicast_address}:{channel_info.port}")
```

After this call, phase-engine logs will show:
```
open_channel: longwire opened 10.000 MHz
open_channel: T3FD opened 10.000 MHz
open_channel: west opened 10.000 MHz
Starting virtual stream for SSRC <n> at 10000000.0 Hz -> 239.x.x.x:5004
```

---

## Summary

| App | Zero-Code Change | Enhanced Integration |
|-----|------------------|---------------------|
| **hf-timestd** | Change `status_address` in config | Pass `reception_mode` per channel |
| **SWL-ka9q** | Change `.radiod-hostname` | Add UI controls for beamforming |
| **Any ka9q app** | Point at phase-engine | Use extended `create_channel()` params |

Phase-engine is designed for **progressive enhancement**: apps work unchanged, and can opt into spatial filtering features as desired.
