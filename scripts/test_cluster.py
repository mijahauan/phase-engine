import subprocess
import time
import signal
import sys
import os

processes = []


def cleanup(sig, frame):
    print("\nShutting down cluster...")
    for p in processes:
        p.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)

# Start 3 mock radios on different multicast status groups
radios = [
    ("radiod-center", "239.192.152.141"),
    ("radiod-north", "239.192.152.142"),
    ("radiod-east", "239.192.152.143"),
]

for name, ip in radios:
    p = subprocess.Popen(["python3", "scripts/mock_radiod.py", "--name", name, "--ip", ip])
    processes.append(p)

print("Cluster started. Press Ctrl+C to stop.")
while True:
    time.sleep(1)
