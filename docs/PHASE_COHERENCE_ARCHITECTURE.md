# Phase Coherence Architecture

Date: 2026-03-01
Version: 0.1.0

This document describes the theoretical and methodological parameters `phase-engine` uses to guarantee absolute phase coherence across an array of independent SDRs (specifically `ka9q-radio` using RX888 hardware).

## The NCO Phase Ambiguity Problem

In modern SDRs, Digital Down Converters (DDCs) use Numerically Controlled Oscillators (NCOs) to mix RF down to baseband IQ. Even when multiple SDRs are disciplined by the exact same 10 MHz GPSDO reference clock, their NCOs do not start simultaneously. 

Because channel creation via network commands (UDP) is asynchronous, an NCO instantiated on `node1` at sample index $N$ will have a completely different mathematical starting phase ($\phi_0$) than an NCO instantiated on `node2` at sample index $N+\Delta$. 

**Crucially:** `ka9q-radio` performs its DDC in software on the host PC, not in the hardware FPGA. Therefore, it lacks a mechanism for "Phase Synchronous Tuning" (i.e., commanding all nodes to reset their NCO phase accumulators to zero on a specific GPS second boundary).

### Methodological Consequence:
**A phase calibration performed at one frequency (e.g., a 900 kHz AM station) cannot be mapped or extrapolated to a different frequency (e.g., 10 MHz WWV).** Every single time a channel is opened, its relative NCO phase is completely randomized.

## Continuous, Per-Channel Phase Alignment

Because global, startup calibration is theoretically invalid for this architecture, `phase-engine` implements **continuous, per-channel alignment**.

1. **Independent Calibration:** When a client (`hf-timestd`) requests a 10 MHz channel, `phase-engine` spawns the 10 MHz streams on all physical nodes.
2. **Data Plane Integration:** Inside the `EgressLoop`, the raw complex samples from all antennas are passed through a sliding-window cross-correlator.
3. **Whole-Block Rotation:** The cross-correlator measures the phase difference of the *entire 24 kHz passband* relative to the reference antenna. Because the software NCO phase error applies uniformly across the entire passband, determining the offset of the dominant carrier allows `phase-engine` to apply a single inverse complex weight $e^{-j\Delta\phi}$ that perfectly aligns all carriers and sub-carriers within that block.
4. **Scintillation Filtering:** Because the target signal (e.g., WWV) is a skywave, its apparent phase scintillates due to ionospheric boiling. The `PhaseCombiner` applies statistical filtering (e.g., a long-time-constant EMA) to the cross-correlation peaks to extract the true underlying hardware NCO offset from the atmospheric noise.

## Group Delay vs. Phase Delay
This architecture explicitly assumes that the physical baseline differences (cable lengths) are small enough that the integer sample delay is zero or negligible across a 24 kHz baseband. If the array baselines grow large enough to induce fractional sample time delays across the passband, simple complex phase rotations will result in group delay distortion.
