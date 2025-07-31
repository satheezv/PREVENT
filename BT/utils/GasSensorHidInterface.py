import hid
import time
from datetime import datetime
from utils.command_ids import GasSensorCommands, SGasConcentration1


class HIDGasSensor:
    """
    Utility class for interacting with the HID Gas Sensor without ROS.
    """

    VENDOR_ID = 0x1FC9  # NXP Vendor ID
    PRODUCT_ID = 0x8251  # Celisca Product ID
    SERIAL_NUMBER = "0024002KMV722ETRKK"  # Your sensor serial number
    WAIT_INTERVAL = 1  # Delay between readings
    VOC_CALIBRATION_FACTOR = 1  # VOC Calibration factor

    def __init__(self):
        """
        Initializes the HID Gas Sensor and establishes a connection.
        """
        self.device = self._connect_device()

    def _connect_device(self):
        """Connects to the HID gas sensor."""
        try:
            dev = hid.device()
            dev.open(self.VENDOR_ID, self.PRODUCT_ID, serial_number=self.SERIAL_NUMBER)
            dev.set_nonblocking(1)
            print(f"Connected to sensor with Serial Number: {self.SERIAL_NUMBER}")
            return dev
        except Exception as e:
            print(f"Error connecting to device: {e}")
            return None

    def send_command(self, command_id):
        """Sends a command to the device and reads the response."""
        if not self.device:
            print("Device not connected")
            return None

        command = [0x00, command_id] + [0] * (64 - 2)  # 64-byte buffer
        self.device.write(command)
        time.sleep(0.1)  # Allow time for response

        data = self.device.read(64)  # Read full 64-byte response
        if not data:
            print("No data received")
            return None

        return data

    def read_gas_data(self):
        """
        Reads gas concentration (VOC & Temperature) from the sensor.
        
        :return: Dictionary containing {"temperature": float, "voc": float} or None if failed.
        """
        raw_data = self.send_command(GasSensorCommands.CommandGetGasConcentration1.value)
        if not raw_data:
            return None

        try:
            gas_data = SGasConcentration1.from_bytes(raw_data)
            return {
                "temperature": gas_data.temperature_c,
                "voc": gas_data.tvoc_ppm * self.VOC_CALIBRATION_FACTOR,
                "iaq": gas_data.iaq,
                "rgas_ohm": gas_data.rgas_ohm,
                "co2": gas_data.co2_ppm
            }
        except ValueError as e:
            print(f"Error processing gas data: {e}")
            return None

    def read_multiple(self, num_reads=1, delay=1):
        """
        Reads gas sensor values multiple times.

        :param num_reads: Number of times to read the data.
        :param delay: Delay in seconds between readings.
        :return: List of sensor readings [{"temperature": float, "voc": float}, ...]
        """
        readings = []
        for _ in range(num_reads):
            gas_data = self.read_gas_data()
            if gas_data:
                print(f"Temperature: {gas_data['temperature']:.2f}°C | VOC: {gas_data['voc']:.1f} PPM")
                readings.append(gas_data)

            time.sleep(delay)

        return readings  # ✅ Returns list of readings instead of writing to CSV

    def close(self):
        """Closes the HID connection."""
        if self.device:
            self.device.close()
            print("Connection closed")
