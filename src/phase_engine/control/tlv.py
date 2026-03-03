"""
TLV decoding for ka9q-radio commands.

Provides a pure-Python decoder for the Type-Length-Value wire format used by
ka9q-radio's control protocol (status.c).  Field type identifiers are imported
from the canonical ``ka9q.types.StatusType`` enum so that phase-engine always
agrees with ka9q-python on tag numbers.

Wire format (per field)::

    [type: 1 byte] [length: 1+ bytes] [value: <length> bytes]

Length uses an extended encoding when the high bit is set (see radiod source).
A type byte of ``StatusType.EOL`` (0) terminates the packet.
"""

import struct
import socket
from typing import Dict, Any

from ka9q.types import StatusType

_FLOAT_TYPES = frozenset({
    StatusType.RADIO_FREQUENCY,
    StatusType.FIRST_LO_FREQUENCY,
    StatusType.SECOND_LO_FREQUENCY,
    StatusType.SHIFT_FREQUENCY,
    StatusType.DOPPLER_FREQUENCY,
    StatusType.DOPPLER_FREQUENCY_RATE,
    StatusType.GAIN,
    StatusType.OUTPUT_LEVEL,
    StatusType.IF_POWER,
    StatusType.BASEBAND_POWER,
    StatusType.NOISE_DENSITY,
    StatusType.LOW_EDGE,
    StatusType.HIGH_EDGE,
    StatusType.HEADROOM,
})

_STRING_TYPES = frozenset({
    StatusType.PRESET,
    StatusType.DESCRIPTION,
})

_SOCKET_TYPES = frozenset({
    StatusType.OUTPUT_DATA_DEST_SOCKET,
    StatusType.OUTPUT_DATA_SOURCE_SOCKET,
    StatusType.STATUS_DEST_SOCKET,
})


def decode_tlv_packet(buffer: bytes) -> Dict[Any, Any]:
    """
    Decode a full ka9q-radio CMD/STATUS packet into a dictionary of {StatusType: value}
    """
    if len(buffer) < 1:
        return {}

    packet_type = buffer[0]
    # We only process CMD (1) or STATUS (0) - upstream expects us to handle CMD
    if packet_type not in (0, 1):
        return {}

    result = {"_packet_type": packet_type}
    cp = 1

    while cp < len(buffer):
        type_val = buffer[cp]
        cp += 1

        if type_val == StatusType.EOL:
            break

        if cp >= len(buffer):
            break

        # Parse length
        optlen = buffer[cp]
        cp += 1

        if optlen & 0x80:
            length_of_length = optlen & 0x7F
            optlen = 0
            for _ in range(length_of_length):
                if cp >= len(buffer):
                    break
                optlen = (optlen << 8) | buffer[cp]
                cp += 1

        if cp + optlen > len(buffer):
            break

        data = buffer[cp : cp + optlen]
        cp += optlen

        # Decode value based on type_val
        val = None

        if type_val in _FLOAT_TYPES:
            if optlen == 4:
                val = struct.unpack(">f", data)[0]
            elif optlen == 8:
                val = struct.unpack(">d", data)[0]
            else:
                val = int.from_bytes(data, byteorder="big", signed=False)
        elif type_val in _STRING_TYPES:
            val = data.decode("utf-8", errors="ignore")
        elif type_val in _SOCKET_TYPES:
            # Wire format: AF_INET (2 bytes) | port (2 bytes) | IPv4 (4 bytes)
            if optlen >= 8:
                port = struct.unpack(">H", data[2:4])[0]
                ip = socket.inet_ntoa(data[4:8])
                val = f"{ip}:{port}"
        # Default Integer
        else:
            if optlen > 0:
                val = int.from_bytes(data, byteorder="big", signed=False)
            else:
                val = 0

        result[type_val] = val

    return result
