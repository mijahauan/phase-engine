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

## Clock Distribution vs RF Feedline Lengths
The physical cabling of the array falls into two entirely separate domains with different impacts on coherence:

1. **The Clock Distribution (GPSDO -> RX888s):**
The GPSDO provides the 27 MHz reference clock to the SDRs. If this clock is distributed via a splitter and cables of **identical length**, the sine wave hits the clock input pin of every ADC at the exact same picosecond. 
This is the optimal hardware configuration. It guarantees zero integer sample delay between the SDRs and ensures they never drift over time ($\Delta\omega = 0$).

2. **The RF Feedlines (Antennas -> RX888s):**
If the coax feedline from Antenna A is 30 meters longer than the feedline from Antenna B, it introduces a physical time delay (group delay) of approximately 150 nanoseconds.
*Does this unequal length break the beamformer?* No, because `phase-engine` processes data in extremely narrow bandwidths (e.g., 24 kHz or 12 kHz). 
The period of a 24 kHz envelope is 41.6 microseconds. A 150 ns group delay across the feedline represents less than 0.4% of a single baseband cycle. Mathematically, this tiny fractional time delay is indistinguishable from a flat **phase shift**. 
When the continuous cross-correlator aligns the array, the measured $\Delta\phi$ it computes is the sum of the software NCO error *and* the static physical feedline delay. By applying a single inverse complex rotation, it zeroes out both errors simultaneously. (Note: if `phase-engine` were processing a wideband 5 MHz block, unequal feedlines *would* cause destructive interference and require fractional-time FIR filters instead of simple phase rotations).

## Physical Antenna Diversity vs. Phased Arrays
The physical nature of the array elements radically changes the DSP strategy required. 

In a traditional **Phased Array**, all antenna elements are physically identical (e.g., a grid of identical vertical whips), have identical polarization, and identical ambient noise floors. In this scenario, combining the signals using Equal Gain Combining (EGC) produces the expected $+10\log_{10}(N)$ gain (e.g., +4.7 dB for 3 antennas) while synthetically steering a null or beam lobe.

However, if the array consists of physically distinct antennas (e.g., a random longwire, a Terminated Folded Dipole, and a vertical dipole), it constitutes a **Diversity Array**.
1. **Gain and Noise Mismatch:** A highly efficient longwire might yield a signal physically 20 dB stronger than a lossy T3FD, but both might share a similar ambient noise floor. Adding them straight across (EGC) would actually *degrade* the array SNR because the weak antenna injects pure noise.
2. **Polarization Diversity:** A horizontal wire and a vertical dipole react entirely differently to the changing polarization of ionospheric skywaves (Faraday rotation). When the horizontal wire enters a deep fade, the vertical dipole often spikes in strength.

### Does Phase Engine Auto-Detect Array Type?
No. The phase engine relies purely on the raw complex voltage of the incoming signals. It does not "know" if the antennas producing those voltages are identical vertical whips or a random assortment of wire antennas. 

Therefore, you must explicitly declare your mathematical intent in `config.toml`:

* **For Diversity Arrays (Mismatched Antennas):** You must set `default_combining_method = "mrc"`. If you mistakenly set it to `mvdr`, the engine will attempt to calculate a spatial steering vector based on your array's XYZ coordinates. Because the phase center and polarization of a longwire vs a vertical are entirely different, this geometric vector will be mathematically invalid, resulting in destructive interference.
* **For Phased Arrays (Identical Antennas):** You can set `default_combining_method = "mvdr"` or `"beamform"`. Because the antennas have identical physical characteristics, the phase difference between them is purely a function of geometric distance and arrival angle. This allows the spatial steering matrix to correctly point beams and carve deep nulls.

### Noise-Normalized Maximum Ratio Combining (MRC)
To optimally exploit a Diversity Array, `phase-engine` utilizes Noise-Normalized Maximum Ratio Combining.
Instead of adding the antennas equally, the DSP calculates an optimal weight for each element: $W_i \propto \frac{S_i}{N_i^2}$, where $S$ is signal voltage and $N^2$ is noise variance.
* When one antenna overwhelmingly dominates (e.g., the longwire on a steady day), the algorithm assigns it ~99% weight, effectively disabling the weaker, noisier antennas to protect the SNR.
* When a polarization fade kills the primary antenna, the continuous alignment algorithm instantaneously detects the SNR shift and transfers the weighting to the vertical antenna that caught the fading wave.
* Thus, for a physically diverse array, the primary benefit of coherence is not raw theoretical beamforming gain, but absolute **fade mitigation** and continuous signal continuity.
