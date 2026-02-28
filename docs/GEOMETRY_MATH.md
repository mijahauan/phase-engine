# Array Geometry & Beamforming Math

Phase Engine implements advanced Digital Signal Processing (DSP) to spatially filter incoming High Frequency (HF) signals. This document explains the mathematical models used to map geographic coordinates to physical phase delays, and how those delays are converted into steering vectors for Minimum Variance Distortionless Response (MVDR) beamforming.

## 1. Geographic Bearing Calculation
When a user requests a focused stream (e.g., `target="WWV"`), the Phase Engine calculates the true geographic bearing from the array's Phase Center (QTH) to the target transmitter.

Given the transmitter coordinates $(\lambda_2, \phi_2)$ and the receiver coordinates $(\lambda_1, \phi_1)$ (where $\lambda$ is latitude and $\phi$ is longitude in radians), the initial bearing $\theta$ (azimuth) is calculated using the haversine bearing formula:

$$ \Delta\phi = \phi_2 - \phi_1 $$
$$ y = \sin(\Delta\phi) \cdot \cos(\lambda_2) $$
$$ x = \cos(\lambda_1) \cdot \sin(\lambda_2) - \sin(\lambda_1) \cdot \cos(\lambda_2) \cdot \cos(\Delta\phi) $$
$$ \theta = \text{atan2}(y, x) $$

This provides the true azimuth in degrees clockwise from Geographic North ($0^\circ$).

## 2. Array Coordinate System (ENU)
The physical antennas must be mapped to a Cartesian coordinate system. Phase Engine uses the East-North-Up (ENU) convention.

* One antenna is designated as the **Phase Center** (Reference Antenna) and assigned coordinate $(0, 0, 0)$.
* All other antennas are mapped relative to this center with coordinates $p_n = [x_n, y_n, z_n]^T$ in meters.
  * $x$-axis: Positive East
  * $y$-axis: Positive North
  * $z$-axis: Positive Up

## 3. The Wave Number Vector
To determine how a plane wave interacts with the physical array, we compute the wave number vector $\vec{k}$.

For a signal arriving **from** azimuth $\theta$ and elevation $\alpha$:
* Frequency: $f$
* Speed of light: $c \approx 299,792,458 \text{ m/s}$
* Wavelength: $\lambda = c / f$
* Wave number: $k = 2\pi / \lambda$

Converting the arrival angles to radians, the Cartesian components of the wave number vector (indicating the direction the wave is traveling) are:

$$ k_x = -k \cdot \sin(\theta) \cdot \cos(\alpha) $$
$$ k_y = -k \cdot \cos(\theta) \cdot \cos(\alpha) $$
$$ k_z = -k \cdot \sin(\alpha) $$

Thus, $\vec{k} = [k_x, k_y, k_z]^T$.

## 4. The Steering Vector
The steering vector $\mathbf{a}$ defines the theoretical complex phase shifts required to perfectly align a signal arriving from $\vec{k}$ across all $N$ antennas.

The geometric phase delay for the $n$-th antenna located at $p_n$ is the dot product of the wave vector and the position vector:

$$ \psi_n = \vec{k} \cdot p_n = (k_x x_n + k_y y_n + k_z z_n) $$

Under the narrowband assumption, the time delay equates directly to a phase shift. The steering vector for the array is:

$$ \mathbf{a}(\theta, \alpha) = \begin{bmatrix} e^{-j\psi_1} \\ e^{-j\psi_2} \\ \vdots \\ e^{-j\psi_N} \end{bmatrix} $$

## 5. DSP Combining Methods

With the steering vector $\mathbf{a}$ and the multi-channel sample matrix $\mathbf{X}$ (dimension $N \times M$ samples), the array can be combined using various weighting strategies $\mathbf{w}$.

The combined output signal $y$ is given by applying the complex conjugate transpose of the weights:
$$ y = \mathbf{w}^H \mathbf{X} $$

### Maximum Ratio Combining (MRC)
MRC is the optimal combiner in a white-noise dominant environment. It weights antennas based on their SNR. The weights are proportional to the cross-correlation vector (channel estimate) relative to the reference antenna.

### Delay-and-Sum Beamforming (Focus)
A pure spatial matched filter. It simply applies the steering vector to co-phase the target signal.
$$ \mathbf{w}_{\text{beamform}} = \frac{\mathbf{a}}{\sqrt{N}} $$

### Minimum Variance Distortionless Response (MVDR)
MVDR (Capon Beamformer) is the algorithm used for the `focus` and `adaptive` reception modes. It ensures unity gain in the direction of the target (distortionless) while actively minimizing total array output power. This mathematically forces deep nulls in the directions of interfering signals.

1. Compute the Spatial Covariance Matrix of the samples:
   $$ \mathbf{R}_{xx} = \frac{1}{M} \mathbf{X} \mathbf{X}^H $$

2. Add diagonal loading to ensure the matrix is invertible (prevents instability with highly correlated signals):
   $$ \mathbf{\tilde{R}}_{xx} = \mathbf{R}_{xx} + \delta \mathbf{I} $$
   *(where $\delta$ is $1\%$ of the mean diagonal power).*

3. Compute the optimal MVDR weights:
   $$ \mathbf{w}_{\text{mvdr}} = \frac{\mathbf{\tilde{R}}_{xx}^{-1} \mathbf{a}}{\mathbf{a}^H \mathbf{\tilde{R}}_{xx}^{-1} \mathbf{a}} $$

This weight vector $\mathbf{w}_{\text{mvdr}}$ is applied instantaneously to the IQ streams to produce the spatially-isolated float32 output stream.
