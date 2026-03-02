"""
RTP Packetizer. Wraps combined IQ data into standard ka9q RTP streams.
"""

import socket
import struct
import logging
import threading
import time
import numpy as np

logger = logging.getLogger(__name__)


class RtpStreamer:
    """Streams combined complex samples out as RTP packets matching ka9q-radio format."""

    def __init__(self, destination_ip: str, port: int, ssrc: int, sample_rate: int):
        self.destination_ip = destination_ip
        self.port = port
        self.ssrc = ssrc
        self.sample_rate = sample_rate

        self.sequence_number = 0
        self.rtp_timestamp = 0

        # Setup socket (UDP Multicast/Unicast)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        
        # Increase the send buffer size to 5MB to handle synchronous multi-channel bursts.
        # This prevents dropped UDP packets on the loopback interface when phase-engine
        # tries to send e.g. 17 channels * 15 packets instantly every 100ms.
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 5 * 1024 * 1024)
        except OSError as e:
            logger.debug(f"Failed to set SO_SNDBUF: {e}")

        # Assuming TTL=2 for local subnets
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    def stream_samples(self, samples: np.ndarray, start_timestamp: int):
        """
        Takes complex64 samples and packetizes them using the provided RTP timestamp.
        
        Args:
            samples: Complex64 array of samples to stream
            start_timestamp: The original GPS-disciplined RTP timestamp of the first sample
            
        ka9q-radio typically sends 320 complex samples per packet for 12kHz/16kHz/24kHz rates.
        Encoding format: F32 (float32 interleaved I, Q)
        """
        SAMPLES_PER_PACKET = 320

        # Sync our internal timestamp tracker with the provided authoritative one.
        # This fixes massive sequence gaps in clients if the egress loop stalls or drops packets.
        self.rtp_timestamp = start_timestamp

        # Flatten complex to float32 interleaved [I0, Q0, I1, Q1, ...]
        flat_samples = np.zeros(len(samples) * 2, dtype=np.float32)
        flat_samples[0::2] = np.real(samples)
        flat_samples[1::2] = np.imag(samples)

        # Packetize
        for i in range(0, len(flat_samples), SAMPLES_PER_PACKET * 2):
            chunk = flat_samples[i : i + SAMPLES_PER_PACKET * 2]
            if len(chunk) == 0:
                continue

            samples_in_chunk = len(chunk) // 2

            # RTP Header (12 bytes)
            # V=2, P=0, X=0, CC=0, M=0, PT=96 (dynamic)
            header = struct.pack(
                "!BBHII",
                0x80,  # V=2 (10), P/X/CC=0 (000000)
                96,  # M=0, PT=96
                self.sequence_number & 0xFFFF,
                self.rtp_timestamp & 0xFFFFFFFF,
                self.ssrc & 0xFFFFFFFF,
            )

            # Payload
            payload = chunk.tobytes()

            # Send
            try:
                self.sock.sendto(header + payload, (self.destination_ip, self.port))
            except OSError as e:
                logger.debug(f"Failed to send RTP packet: {e}")

            self.sequence_number = (self.sequence_number + 1) % 65536
            self.rtp_timestamp = (self.rtp_timestamp + samples_in_chunk) % (2**32)

    def close(self):
        if self.sock:
            self.sock.close()
