#!/usr/bin/env python3
"""
Mock Radiod Generator

Simulates a physical radiod receiver by multicasting dummy RTP IQ data and
listening on the status port. Used for integration testing phase-engine locally.
"""

import socket
import struct
import time
import json
import threading
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

MCAST_GRP = "239.192.152.141"
MCAST_PORT = 5006
DATA_PORT = 5004


def get_status_json(name, channels=None):
    return json.dumps(
        {"name": name, "state": "running", "backend": "mock", "channels": channels or {}}
    ).encode("utf-8")


@dataclass
class MockChannel:
    ssrc: int
    frequency: float
    destination: str
    sample_rate: int
    running: bool = True


class MockRadiod:
    def __init__(self, name="mock-radiod", status_ip=MCAST_GRP, status_port=MCAST_PORT):
        self.name = name
        self.status_ip = status_ip
        self.status_port = status_port

        # Multicast state socket
        self.status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.status_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

        # Unicast control socket (listen on any interface, port 5006)
        self.ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            # Bind to the specific IP to avoid conflicts when running multiple mocks on loopback
            self.ctrl_sock.bind((status_ip, 5006))
        except OSError:
            # Fallback
            self.ctrl_sock.bind(("0.0.0.0", 5006))

        # RTP Egress socket
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.running = False
        self.channels: Dict[int, MockChannel] = {}
        self.next_ssrc = 1000

    def _status_loop(self):
        while self.running:
            try:
                # ka9q-python's discover_channels_native expects packet[0] == 0 (STATUS type)
                for ssrc, ch in self.channels.items():
                    buf = bytearray()
                    buf.append(0)  # Packet Type: STATUS

                    # Tag 18: OUTPUT_SSRC (uint32)
                    # We must compress leading zeros for integers just like ka9q-python does!
                    ssrc_bytes = ssrc.to_bytes(4, byteorder="big").lstrip(b"\x00")
                    if not ssrc_bytes:
                        buf.extend([18, 0])
                    else:
                        buf.extend([18, len(ssrc_bytes)])
                        buf.extend(ssrc_bytes)
                    # Tag 1: FREQ (float64)
                    buf.extend([1, 8])
                    buf.extend(struct.pack(">d", ch.frequency))

                    # Tag 2: PRESET (string)
                    preset_bytes = b"iq" + b"\x00"
                    buf.extend([2, len(preset_bytes)])
                    buf.extend(preset_bytes)

                    # Tag 3: SAMPLE_RATE (int32)
                    sr_bytes = ch.sample_rate.to_bytes(4, byteorder="big").lstrip(b"\x00")
                    if not sr_bytes:
                        buf.extend([3, 0])
                    else:
                        buf.extend([3, len(sr_bytes)])
                        buf.extend(sr_bytes)
                    # Tag 17: OUTPUT_DATA_DEST_SOCKET (ka9q-python expects this for 'destination')
                    ip = ch.destination.split(":")[0] if ":" in ch.destination else "127.0.0.1"
                    port = int(ch.destination.split(":")[1]) if ":" in ch.destination else 5004

                    buf.extend([17, 8])
                    buf.extend(struct.pack(">H", socket.AF_INET))
                    buf.extend(struct.pack(">H", port))
                    buf.extend(socket.inet_aton(ip))

                    # Tag 21: RTP_DEST (string)
                    dest_bytes = ch.destination.encode("utf-8") + b"\x00"
                    buf.extend([21, len(dest_bytes)])
                    buf.extend(dest_bytes)
                    # Tag 0: EOL
                    buf.extend([0, 0])

                    self.status_sock.sendto(buf, (self.status_ip, self.status_port))

                time.sleep(0.5)
            except Exception as e:
                print(f"Error sending status: {e}")

    def _parse_tlv(self, data: bytes) -> Dict[str, Any]:
        """Very basic TLV parser for ka9q control commands."""
        if len(data) < 1:
            return {}

        if data[0] != 1:  # CMD packet type
            return {}

        offset = 1
        params = {}
        while offset < len(data):
            type_val = data[offset]
            offset += 1

            if type_val == 0:  # EOL
                break

            if offset >= len(data):
                break

            optlen = data[offset]
            offset += 1
            if optlen & 0x80:
                length_of_length = optlen & 0x7F
                optlen = 0
                for _ in range(length_of_length):
                    if offset >= len(data):
                        break
                    optlen = (optlen << 8) | data[offset]
                    offset += 1

            if offset + optlen > len(data):
                break

            val = data[offset : offset + optlen]

            try:
                if type_val == 1 and optlen == 8:
                    params["freq"] = struct.unpack(">d", val)[0]
                elif type_val == 2:
                    params["preset"] = val.decode("utf-8").replace("\x00", "")
                elif type_val == 3:
                    ival = 0
                    for b in val:
                        ival = (ival << 8) | b
                    params["rate"] = ival
                elif type_val == 18:
                    # OUTPUT_SSRC
                    ival = 0
                    for b in val:
                        ival = (ival << 8) | b
                    params["ssrc"] = ival
                elif type_val == 21:
                    params["dest"] = val.decode("utf-8").replace("\x00", "")
                elif type_val == 17:
                    # DEST_SOCKET (family, port, ip)
                    if optlen == 8:
                        family, port = struct.unpack(">HH", val[:4])
                        ip = socket.inet_ntoa(val[4:8])
                        params["dest"] = f"{ip}:{port}"
            except Exception:
                pass

            offset += optlen

        return params

    def _control_loop(self):
        addr = self.ctrl_sock.getsockname()
        print(f"[{self.name}] Listening for TLV control commands on {addr}")
        while self.running:
            try:
                self.ctrl_sock.settimeout(1.0)
                data, remote_addr = self.ctrl_sock.recvfrom(2048)

                params = self._parse_tlv(data)
                if not params:
                    continue

                # ka9q-python sends separate packets for frequency, preset, etc.,
                # but all contain the SSRC.
                ssrc = params.get("ssrc")
                if ssrc:
                    # Force deterministic internal allocation if missing
                    if ssrc is None:
                        import hashlib

                        # simplified mimic of allocate_ssrc to keep channels consistent across mocks
                        s = f"{params.get('freq', 10e6):.0f}-{params.get('preset', 'iq')}-{params.get('rate', 12000)}"
                        h = hashlib.md5(s.encode()).digest()
                        ssrc = (h[0] << 24 | h[1] << 16 | h[2] << 8 | h[3]) & 0x7FFFFFFF

                    if ssrc not in self.channels:
                        # create
                        self.channels[ssrc] = MockChannel(
                            ssrc=ssrc,
                            frequency=params.get("freq", 10e6),
                            destination=params.get("dest", f"{self.status_ip}:5004"),
                            sample_rate=params.get("rate", 12000) or 12000,
                        )
                        print(f"[{self.name}] Created channel {ssrc}")
                    else:
                        # update
                        ch = self.channels[ssrc]
                        if "freq" in params:
                            ch.frequency = params["freq"]
                        if "rate" in params:
                            ch.sample_rate = params["rate"]
                        if "dest" in params:
                            ch.destination = params["dest"]

            except socket.timeout:
                continue
            except Exception as e:
                print(f"[{self.name}] Control error: {e}")

    def _parse_tlv(self, data: bytes) -> Dict[str, Any]:
        """Very basic TLV parser for ka9q control commands."""
        if len(data) < 1:
            return {}

        if data[0] != 1:  # CMD packet type
            return {}

        offset = 1
        params = {}
        while offset < len(data):
            type_val = data[offset]
            offset += 1

            if type_val == 0:  # EOL
                break

            if offset >= len(data):
                break

            optlen = data[offset]
            offset += 1
            if optlen & 0x80:
                length_of_length = optlen & 0x7F
                optlen = 0
                for _ in range(length_of_length):
                    if offset >= len(data):
                        break
                    optlen = (optlen << 8) | data[offset]
                    offset += 1

            if offset + optlen > len(data):
                break

            val = data[offset : offset + optlen]

            try:
                if type_val == 1 and optlen == 8:
                    params["freq"] = struct.unpack(">d", val)[0]
                elif type_val == 2:
                    params["preset"] = val.decode("utf-8").replace("\x00", "")
                elif type_val == 3:
                    ival = 0
                    for b in val:
                        ival = (ival << 8) | b
                    params["rate"] = ival
                elif type_val == 18:
                    # OUTPUT_SSRC
                    ival = 0
                    for b in val:
                        ival = (ival << 8) | b
                    params["ssrc"] = ival
                elif type_val == 21:
                    params["dest"] = val.decode("utf-8").replace("\x00", "")
                elif type_val == 17:
                    # DEST_SOCKET (family, port, ip)
                    if optlen == 8:
                        family, port = struct.unpack(">HH", val[:4])
                        ip = socket.inet_ntoa(val[4:8])
                        params["dest"] = f"{ip}:{port}"
            except Exception:
                pass

            offset += optlen

        return params

    def _control_loop(self):
        addr = self.ctrl_sock.getsockname()
        print(f"[{self.name}] Listening for TLV control commands on {addr}")
        while self.running:
            try:
                self.ctrl_sock.settimeout(1.0)
                data, remote_addr = self.ctrl_sock.recvfrom(2048)

                params = self._parse_tlv(data)
                if not params:
                    continue

                # ka9q-python sends separate packets for frequency, preset, etc.,
                # but all contain the SSRC.
                ssrc = params.get("ssrc")
                if ssrc:
                    # Force deterministic internal allocation if missing
                    if ssrc is None:
                        import hashlib

                        # simplified mimic of allocate_ssrc to keep channels consistent across mocks
                        s = f"{params.get('freq', 10e6):.0f}-{params.get('preset', 'iq')}-{params.get('rate', 12000)}"
                        h = hashlib.md5(s.encode()).digest()
                        ssrc = (h[0] << 24 | h[1] << 16 | h[2] << 8 | h[3]) & 0x7FFFFFFF

                    if ssrc not in self.channels:
                        # create
                        self.channels[ssrc] = MockChannel(
                            ssrc=ssrc,
                            frequency=params.get("freq", 10e6),
                            destination=params.get("dest", f"{self.status_ip}:5004"),
                            sample_rate=params.get("rate", 12000) or 12000,
                        )
                        print(f"[{self.name}] Created channel {ssrc}")
                    else:
                        # update
                        ch = self.channels[ssrc]
                        if "freq" in params:
                            ch.frequency = params["freq"]
                        if "rate" in params:
                            ch.sample_rate = params["rate"]
                        if "dest" in params:
                            ch.destination = params["dest"]

            except socket.timeout:
                continue
            except Exception as e:
                print(f"[{self.name}] Control error: {e}")

    def _data_loop(self):
        samples_per_packet = 320
        seq = 0
        timestamp = 0

        while self.running:
            time.sleep(0.026)

            for ssrc, ch in list(self.channels.items()):
                if not ch.running:
                    continue

                t = np.arange(samples_per_packet) / ch.sample_rate
                tone = np.exp(1j * 2 * np.pi * 1000 * t).astype(np.complex64)
                noise = (
                    np.random.randn(samples_per_packet) + 1j * np.random.randn(samples_per_packet)
                ) * 0.1
                samples = (tone + noise).astype(np.complex64)

                # F32 Encoding format from ka9q-radio
                header = struct.pack(">BBHII", 0x80, 96, seq, timestamp, ssrc)
                packet = header + samples.tobytes()

                try:
                    ip, port = ch.destination.split(":")
                    self.data_sock.sendto(packet, (ip, int(port)))
                except Exception:
                    pass

            seq = (seq + 1) % 65536
            timestamp += samples_per_packet

    def start(self):
        self.running = True
        self.status_thread = threading.Thread(target=self._status_loop, daemon=True)
        self.ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.data_thread = threading.Thread(target=self._data_loop, daemon=True)

        self.status_thread.start()
        self.ctrl_thread.start()
        self.data_thread.start()
        print(f"[{self.name}] Started status multicaster on {self.status_ip}:{self.status_port}")

    def stop(self):
        self.running = False
        if hasattr(self, "status_thread"):
            self.status_thread.join()
        if hasattr(self, "ctrl_thread"):
            self.ctrl_thread.join()
        if hasattr(self, "data_thread"):
            self.data_thread.join()
        self.ctrl_sock.close()
        self.status_sock.close()
        self.data_sock.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="mock-radiod-1")
    parser.add_argument("--ip", default="239.192.152.141")
    parser.add_argument("--port", type=int, default=5006)
    args = parser.parse_args()

    mock = MockRadiod(name=args.name, status_ip=args.ip, status_port=args.port)
    try:
        mock.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        mock.stop()
