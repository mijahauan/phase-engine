#!/usr/bin/env python3
"""
Phase Engine Source Monitor

Connects to the phase-engine proxy to discover all currently 
active virtual channels, and maps them back to the underlying 
physical SDR sources to report their health and packet flow.
"""

import argparse
import time
import socket
import struct
import numpy as np
from datetime import datetime

from ka9q.discovery import discover_channels

def get_engine_status(status_address: str):
    """
    Since the CLI status endpoint isn't fully wired yet, 
    we discover the active channels via the standard ka9q 
    multicast discovery on the phase-engine's status address.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Querying Phase Engine at {status_address}...")
    try:
        channels = discover_channels(status_address, listen_duration=1.0)
        return channels
    except Exception as e:
        print(f"Failed to query phase-engine: {e}")
        return {}

def monitor(status_address: str, duration: int):
    print("="*60)
    print("PHASE ENGINE SOURCE MONITOR")
    print("="*60)
    
    start_time = time.time()
    
    while True:
        if duration > 0 and (time.time() - start_time) > duration:
            break
            
        channels = get_engine_status(status_address)
        
        if not channels:
            print("No active virtual channels found. Engine may be idle or offline.")
        else:
            print(f"\nActive Virtual Channels: {len(channels)}")
            print("-" * 60)
            print(f"{'V-SSRC':<12} | {'Freq (MHz)':<10} | {'Output Addr':<22} | {'Sample Rate'}")
            print("-" * 60)
            
            # Sort channels by output address (IP:Port), then by Frequency
            sorted_channels = sorted(
                channels.items(),
                key=lambda x: (f"{x[1].multicast_address}:{x[1].port}", x[1].frequency)
            )
            
            for ssrc, c in sorted_channels:
                freq_mhz = c.frequency / 1e6
                addr = f"{c.multicast_address}:{c.port}"
                print(f"{ssrc:<12} | {freq_mhz:<10.3f} | {addr:<22} | {c.sample_rate} Hz")
                
            print("-" * 60)
            print("To view physical source health, check the engine journal:")
            print("  sudo journalctl -u phase-engine -f")
            
        if duration > 0:
            time.sleep(2.0)
        else:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase Engine Source Monitor")
    parser.add_argument("--status", default="239.99.1.1", help="Phase Engine status multicast address")
    parser.add_argument("--continuous", "-c", action="store_true", help="Run continuously")
    args = parser.parse_args()
    
    monitor(args.status, 86400 if args.continuous else 0)
