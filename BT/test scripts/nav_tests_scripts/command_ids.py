from enum import Enum
import struct
from dataclasses import dataclass

# USB Buffer Size
USB_BUFFER_SIZE = 64  # Same as in C++

# --- 🚀 Command IDs Enum ---
class GasSensorCommands(Enum):
    CommandStatus = 1
    CommandVersion = 2
    CommandGetGasConcentration1 = 3
    CommandSetSensorEnable = 4
    CommandGetSensorEnable = 5
    CommandSetParameters1 = 6
    CommandGetParameters1 = 7
    CommandSetClock = 8
    CommandGetClock = 9
    CommandGetHiResPressure = 10
    CommandGetGasConcentration2 = 11
    CommandSetParameters2 = 12
    CommandGetParameters2 = 13
    CommandGetGasConcentration3 = 14
    CommandGetDistance = 15
    CommandSwitchLeds = 16

    # Seminar-specific commands
    CommandGetSeminarTemperature = 235
    CommandGetSeminarData = 236

    # Beacon & WLAN Commands
    CommandGetBeaconData = 240
    CommandSetBeaconParameters = 241
    CommandGetBeaconParameters = 242

    CommandGetWlanSsidParameters = 250
    CommandSetWlanSsidParameters = 251
    CommandGetWlanParameters = 252
    CommandSetWlanParameters = 253

    CommandBootLoader = 254
    CommandTestByteAlignment = 255


# --- 🚀 Error Codes Enum ---
class GasSensorError(Enum):
    NoError = 0
    ErrorUSBCommandFailed = 1
    ErrorUSBCommandPending = 2
    ErrorUnknownCommand = 3


# --- 🚀 Data Structures (Converted from C++ Structs) ---

@dataclass
class SGasConcentration1:
    """Gas concentration structure (BME680 + SGP30)."""
    command_name: int
    return_code: int
    bme680_chip_id: int
    co2_ppm: float
    tvoc_ppm: float
    temperature_c: float
    humidity_percent: float
    pressure_mbar: float
    iaq: float
    rgas_ohm: float

    @staticmethod
    def from_bytes(data):
        """Unpacks sensor data from 64-byte HID response."""
        if not isinstance(data, (bytes, bytearray)):
            data = bytes(data)  # 🔥 Convert list to bytes before unpacking

        if len(data) < 32:  # 🔥 Ensure we have enough bytes
            raise ValueError(f"Expected at least 32 bytes, but got {len(data)}")

        unpacked = struct.unpack("<BBHfffffff", data[:32])  # Adjusted to 32 bytes
        return SGasConcentration1(*unpacked)

