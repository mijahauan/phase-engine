"""
TLV decoding for ka9q-radio commands.
Provides a pure Python port of radiod's decode_X functions from status.c
"""

import struct
import socket
from typing import Dict, Any


# StatusType enumeration (matching ka9q/types.py and status.h)
class StatusType:
    EOL = 0
    COMMAND_TAG = 1
    CMD_CNT = 2
    GPS_TIME = 3
    DESCRIPTION = 4
    STATUS_DEST_SOCKET = 5
    SETOPTS = 6
    CLEAROPTS = 7
    RTP_TIMESNAP = 8
    BIN_BYTE_DATA = 9
    INPUT_SAMPRATE = 10

    OUTPUT_DATA_SOURCE_SOCKET = 16
    OUTPUT_DATA_DEST_SOCKET = 17
    OUTPUT_SSRC = 18
    OUTPUT_TTL = 19
    OUTPUT_SAMPRATE = 20

    RADIO_FREQUENCY = 33

    AGC_ENABLE = 52
    GAIN = 56
    OUTPUT_LEVEL = 57
    OUTPUT_SAMPLES = 58

    PRESET = 68

    OUTPUT_ENCODING = 82


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
        # Float/Double
        if type_val in (StatusType.RADIO_FREQUENCY, StatusType.GAIN, StatusType.OUTPUT_LEVEL):
            if optlen == 4:
                val = struct.unpack(">f", data)[0]
            elif optlen == 8:
                val = struct.unpack(">d", data)[0]
            else:
                val = int.from_bytes(data, byteorder="big", signed=False)
        # String
        elif type_val in (StatusType.PRESET, StatusType.DESCRIPTION):
            val = data.decode("utf-8", errors="ignore")
        # Sockets
        elif type_val in (StatusType.OUTPUT_DATA_DEST_SOCKET, StatusType.STATUS_DEST_SOCKET):
            if optlen >= 6:
                ip = socket.inet_ntoa(data[:4])
                port = struct.unpack(">H", data[4:6])[0]
                val = f"{ip}:{port}"
        # Default Integer
        else:
            if optlen > 0:
                val = int.from_bytes(data, byteorder="big", signed=False)
            else:
                val = 0

        result[type_val] = val

    return result
