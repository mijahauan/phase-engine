import numpy as np
import matplotlib.pyplot as plt
import math

from phase_engine.dsp.array_geometry import AntennaArray
from phase_engine.dsp.combiner import PhaseCombiner
from phase_engine.engine import calculate_bearing

def plot_array_pattern(freq_hz: float, qth_lat: float, qth_lon: float, target_lat: float, target_lon: float, target_name: str, interferer_lat: float = None, interferer_lon: float = None, interferer_name: str = None):
    # Setup our 3-element test array (Center, North, East)
    # Using approx 15m spacing (lambda/2 at 10 MHz)
    positions = {
        "center": (0.0, 0.0, 0.0),
        "north": (0.0, 15.0, 0.0),
        "east": (15.0, 0.0, 0.0)
    }
    array = AntennaArray("center", positions)
    
    # Calculate bearing to target
    target_bearing = calculate_bearing(qth_lat, qth_lon, target_lat, target_lon)
    print(f"Target {target_name} Bearing: {target_bearing:.1f}°")
    
    # Get the steering vector for the target
    a_target = array.get_steering_vector(freq_hz, target_bearing)
    
    # We'll use standard delay-and-sum beamforming weights for the basic pattern
    w = a_target / np.sqrt(array.n_elements)
    
    # If we have an interferer, we can calculate MVDR weights
    if interferer_lat is not None:
        int_bearing = calculate_bearing(qth_lat, qth_lon, interferer_lat, interferer_lon)
        print(f"Interferer {interferer_name} Bearing: {int_bearing:.1f}°")
        
        # Simulate a covariance matrix with strong signal from target and interferer
        # Rxx = a_target * a_target^H + P_int * a_int * a_int^H + noise
        a_int = array.get_steering_vector(freq_hz, int_bearing)
        
        # Power levels (linear)
        P_target = 10.0
        P_int = 100.0 # Strong interferer
        noise_var = 1.0
        
        Rxx = P_target * np.outer(a_target, a_target.conj()) + \
              P_int * np.outer(a_int, a_int.conj()) + \
              noise_var * np.eye(array.n_elements)
              
        # Calculate MVDR weights manually (combiner expects raw samples, not Rxx)
        Rxx_inv = np.linalg.inv(Rxx + 0.01 * np.eye(array.n_elements))
        num = Rxx_inv @ a_target
        den = a_target.conj().T @ Rxx_inv @ a_target
        w_mvdr = num / den
        w = w_mvdr # Use MVDR weights for the plot
        
        title = f"{target_name} ({target_bearing:.0f}°) MVDR Pattern w/ {interferer_name} Null ({int_bearing:.0f}°)"
    else:
        title = f"{target_name} ({target_bearing:.0f}°) Delay-and-Sum Pattern"
        
    # Calculate response across all 360 degrees
    angles = np.linspace(0, 360, 360)
    response = np.zeros_like(angles, dtype=complex)
    
    for i, angle in enumerate(angles):
        # The response at any angle is w^H * a(angle)
        a_theta = array.get_steering_vector(freq_hz, angle)
        response[i] = w.conj().T @ a_theta
        
    # Convert to power in dB
    power = np.abs(response)**2
    power_db = 10 * np.log10(power + 1e-10) # Avoid log(0)
    
    # Normalize so peak is 0 dB
    power_db = power_db - np.max(power_db)
    
    # Ensure -40dB floor for plotting
    power_db = np.maximum(power_db, -40)
    
    # Plotting
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # Matplotlib polar plots are counter-clockwise from East by default.
    # We want clockwise from North (standard compass bearings)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Plot the pattern
    theta_rad = np.radians(angles)
    ax.plot(theta_rad, power_db)
    ax.fill(theta_rad, power_db, alpha=0.3)
    
    # Add target marker
    ax.plot(math.radians(target_bearing), 0, 'go', markersize=10, label=f'Target ({target_name})')
    
    if interferer_lat is not None:
        ax.plot(math.radians(int_bearing), 0, 'ro', markersize=10, label=f'Null ({interferer_name})')
        
    ax.set_rlabel_position(-22.5)  # Move grid labels away from center
    ax.set_rticks([-40, -30, -20, -10, 0])
    ax.set_title(title + f"\n{freq_hz/1e6:.1f} MHz, 3-Element 'L' Array", va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.savefig(f"/tmp/pattern_{target_name.replace(' ', '_')}.png", bbox_inches='tight')
    print(f"Saved plot to /tmp/pattern_{target_name.replace(' ', '_')}.png")

if __name__ == "__main__":
    # Example Coordinates
    # QTH: KC (approx)
    qth_lat = 39.0997
    qth_lon = -94.5786
    
    # WWV: Fort Collins, CO
    wwv_lat = 40.6780
    wwv_lon = -105.0470
    
    # CHU: Ottawa, ON
    chu_lat = 45.2952
    chu_lon = -75.7533
    
    # BPM: Pucheng, China
    bpm_lat = 34.9500
    bpm_lon = 109.5330
    
    # 1. Plot basic pattern targeting WWV
    plot_array_pattern(10e6, qth_lat, qth_lon, wwv_lat, wwv_lon, "WWV")
    
    # 2. Plot MVDR pattern targeting WWV, nulling BPM (which often shares 10 MHz)
    plot_array_pattern(10e6, qth_lat, qth_lon, wwv_lat, wwv_lon, "WWV", bpm_lat, bpm_lon, "BPM")
    
    # 3. Plot pattern targeting CHU (3.33 MHz)
    plot_array_pattern(3.33e6, qth_lat, qth_lon, chu_lat, chu_lon, "CHU")
