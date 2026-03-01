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

## Why Shared GPSDO is Not Enough
It is a common misconception that disciplining all SDRs with a shared 10 MHz/27 MHz GPSDO clock guarantees phase alignment. 

A shared GPSDO guarantees **Frequency Synchronization** ($\Delta\omega = 0$). This ensures the sample rates are identical and the ADCs sample at the exact same picosecond. Without this, the array would immediately collapse as the relative phase spins continuously like a propeller $e^{-j(\Delta\omega)t}$.

However, the DDC process mixes the RF down using a software-generated sine wave. The starting phase of that sine wave, $\phi_0$, is determined by the exact microsecond the software thread initializes. Because the SDRs share a GPSDO, the relative phase offset $\Delta\phi = \phi_{0, node1} - \phi_{0, node2}$ will remain perfectly locked and constant forever. But the **initial value** of that constant is a completely unknown random variable $[0, 2\pi)$.

## Handling Multiple Carriers in a Shared Channel
When `hf-timestd` requests a channel (e.g., 24 kHz wide centered at 10 MHz), that block contains WWV's carrier, its AM subcarriers, and potentially interfering stations. 

How can `phase-engine` continuously align the array if there are multiple signals arriving from different angles?

The software DDC applies its random NCO phase error uniformly to the **entire 24 kHz block**. Therefore:
1. `phase-engine` cross-correlates the full 24 kHz complex stream between `node1` and `node2`.
2. The dominant energy in the passband (usually the main carrier) generates a massive peak in the cross-correlation matrix.
3. Once the algorithm calculates the offset of that peak, it applies the inverse rotation $e^{-j\Delta\phi}$ to the **entire 24 kHz stream**.
4. Because the NCO error was applied uniformly to the block, rotating the whole block perfectly phase-aligns every single sub-carrier and signal within it simultaneously.

## EMA Smoothing and Quality Control
Because skywave signals fade and scintillate, a single bad correlation frame could ruin the beamforming. 
To prevent this, `phase-engine` applies an Exponential Moving Average (EMA) filter (e.g., $\alpha = 0.05$) to the complex correlation weights. 

If the correlation score drops below a mathematical threshold (e.g., during a deep fade), the engine freezes the weights at their last known good state, waiting for the signal to return, rather than spinning the array based on noise.
