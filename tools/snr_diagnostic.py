#!/usr/bin/env python3
"""
Diagnostic tool to verify mathematical phase coherence, measure SNR gain,
and detect frequency/clock offsets between SDRs in the array.
"""

import numpy as np
import time
import socket
import struct
import logging
import argparse
import hashlib
from scipy import signal
from ka9q.discovery import discover_channels

logging.basicConfig(level=logging.ERROR)

def get_ssrc_base(name):
    h = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
    return (h & 0x7FFF0000) | 0x00010000

def capture_udp(ip, port, ssrc=None, duration=5.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', port))
    
    mreq = struct.pack("4sl", socket.inet_aton(ip), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(0.5)
    
    start = time.time()
    samples = []
    while time.time() - start < duration:
        try:
            data, _ = sock.recvfrom(8192)
            if len(data) < 12: continue
            
            pkt_ssrc = struct.unpack("!I", data[8:12])[0]
            if ssrc is not None and pkt_ssrc != ssrc:
                continue
                
            payload = data[12:]
            s = np.frombuffer(payload, dtype=np.float32).view(np.complex64)
            s_real = np.nan_to_num(np.real(s))
            s_imag = np.nan_to_num(np.imag(s))
            s = s_real + 1j * s_imag
            samples.append(s.astype(np.complex64))
        except socket.timeout:
            continue
    return np.concatenate(samples) if samples else np.array([])

def get_snr(s, sample_rate=12000):
    if len(s) == 0: return 0.0
    
    spn_power = np.var(s)
    f, Pxx = signal.welch(s, fs=sample_rate, nperseg=4096, return_onesided=False, scaling='density')
    n0 = np.percentile(Pxx, 10)
    noise_10k = n0 * 10000
    
    signal_power = spn_power - noise_10k
    if signal_power <= 0: signal_power = 1e-12
    
    return 10 * np.log10(signal_power / noise_10k)

def get_carrier_offset(s, sample_rate=12000):
    if len(s) == 0: return 0.0
    NFFT = min(16384, len(s))
    f, Pxx = signal.welch(s, fs=sample_rate, nperseg=NFFT, window='boxcar', return_onesided=False, scaling='spectrum')
    f_shifted = np.fft.fftshift(f)
    Pxx_shifted = np.fft.fftshift(Pxx)
    peak_idx = np.argmax(Pxx_shifted)
    return f_shifted[peak_idx]

def analyze_array_performance(raw_list, names, sample_rate=12000):
    print("\n" + "="*60)
    print("PHASE ENGINE HARDWARE DIAGNOSTIC REPORT")
    print("="*60)
    
    window = min([len(s) for s in raw_list])
    if window == 0:
        print("ERROR: No data captured.")
        return
        
    X = np.zeros((len(raw_list), window), dtype=np.complex64)
    for i in range(len(raw_list)):
        X[i, :] = raw_list[i][:window]
        
    print("\n--- CARRIER FREQUENCY & CLOCK ALIGNMENT ---")
    print("For phase coherence, all carriers must be exactly at the same frequency.")
    
    freqs = []
    snrs = []
    
    for i in range(len(raw_list)):
        s = X[i, :]
        peak_freq = get_carrier_offset(s, sample_rate)
        freqs.append(peak_freq)
        
        snr_db = get_snr(s, sample_rate)
        snrs.append(snr_db)
        
        print(f"{names[i]:<15} | Peak offset: {peak_freq:>7.1f} Hz | SNR: {snr_db:>5.1f} dB")
        
    freq_diffs = [abs(f - freqs[0]) for f in freqs[1:]]
    max_diff = max(freq_diffs) if freq_diffs else 0.0
    
    print("\n--- NOISE-NORMALIZED MAXIMUM RATIO COMBINING ---")
    
    # If SDRs have wildly different analog gain, we must normalize the noise floor
    # before calculating cross-correlation, otherwise the loudest SDR overrides MRC entirely.
    
    X_norm = np.zeros_like(X)
    for i in range(len(raw_list)):
        # Normalize to unit variance
        variance = np.var(X[i, :])
        if variance > 0:
            X_norm[i, :] = X[i, :] / np.sqrt(variance)
        else:
            X_norm[i, :] = X[i, :]
            
    ref_signal = X_norm[0, :]
    cross_corr = np.mean(X_norm * ref_signal.conj(), axis=1)
    
    norm = np.linalg.norm(cross_corr)
    if norm > 0:
        weights = cross_corr / norm
    else:
        weights = np.zeros(len(raw_list), dtype=np.complex64)
        weights[0] = 1.0
        
    for i in range(len(raw_list)):
        print(f"{names[i]:<15} | Weight Mag: {np.abs(weights[i]):.4f} | Phase: {np.degrees(np.angle(weights[i])):6.1f}°")
        
    combined_mrc = np.dot(weights.conj(), X_norm)
    s_mrc = get_snr(combined_mrc, sample_rate)
    
    s_sel = max(snrs)
    
    print(f"\nSelection Diversity (Best Ant): {s_sel:5.1f} dB")
    print(f"Maximum Ratio Combining:        {s_mrc:5.1f} dB")
    print(f"Hardware Array Gain:            {s_mrc - s_sel:+.1f} dB")
    
    print("\n" + "="*60)
    if max_diff > 1.0:
        print("CRITICAL ERROR: SDR CLOCK DESYNCHRONIZATION DETECTED")
        print("Your SDRs are operating at different physical frequencies.")
        print(f"Maximum frequency drift detected: {max_diff:.1f} Hz.")
        print("A frequency offset causes the phase between antennas to continuously rotate.")
        print("Because the phase is spinning rapidly, the mathematical cross-correlation")
        print("averages to zero. The phase-engine defensive logic correctly assigns a weight")
        print("of ~0.0 to the drifting antennas to prevent destructive interference.")
    elif (s_mrc - s_sel) > 0.5:
        print("SUCCESS: PHASE COHERENCE ACHIEVED")
    else:
        print("CONCLUSION: Array is frequency-locked, but SNR gain is low.")
        print("This is likely due to highly correlated local noise or disparate SNR levels.")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase Engine SNR Diagnostic")
    parser.add_argument("--freq", type=float, default=10e6, help="Frequency in Hz (default: 10e6)")
    parser.add_argument("--duration", type=float, default=5.0, help="Capture duration in seconds (default: 5.0)")
    
    parser.add_argument("--ant1", type=str, default="longwire,bee3-status.local", help="name,status_address")
    parser.add_argument("--ant2", type=str, default="T3FD,bee1-status.local", help="name,status_address")
    parser.add_argument("--ant3", type=str, default="west,bee4-status.local", help="name,status_address")
    
    args = parser.parse_args()
    
    print(f"Discovering physical antennas at {args.freq/1e6} MHz...")
    
    ants = [args.ant1, args.ant2, args.ant3]
    hosts = []
    for a in ants:
        if "," in a:
            name, status = a.split(",", 1)
            hosts.append((name, status))
            
    found_channels = []
    sample_rate = 12000
    
    for name, status_host in hosts:
        base = get_ssrc_base(name)
        try:
            chans = discover_channels(status_host, listen_duration=0.5)
            for c in chans.values():
                if c.frequency == args.freq and (c.ssrc & 0xFFFF0000) == (base & 0xFFFF0000):
                    found_channels.append((name, c))
                    sample_rate = c.sample_rate
                    break
        except Exception as e:
            pass
            
    if len(found_channels) < 2:
        print(f"Need at least 2 active physical streams. Found {len(found_channels)}.")
        exit(1)
        
    import threading
    class CaptureThread(threading.Thread):
        def __init__(self, ip, port, ssrc):
            super().__init__()
            self.ip = ip
            self.port = port
            self.ssrc = ssrc
            self.data = np.array([])
        def run(self):
            self.data = capture_udp(self.ip, self.port, ssrc=self.ssrc, duration=args.duration)

    print(f"\nCapturing {args.duration} seconds of independent physical array data from {len(found_channels)} antennas...")
    threads = []
    names = []
    for name, c in found_channels:
        names.append(name)
        t = CaptureThread(c.multicast_address, c.port, c.ssrc)
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    valid_data = [t.data for t in threads if len(t.data) > 0]
    valid_names = [names[i] for i, t in enumerate(threads) if len(t.data) > 0]
    
    if len(valid_data) >= 2:
        analyze_array_performance(valid_data, valid_names, sample_rate)
    else:
        print("Failed to capture data from at least 2 antennas.")
