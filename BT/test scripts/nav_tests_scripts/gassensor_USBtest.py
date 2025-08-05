import hid
import time
import csv
from datetime import datetime
import pandas as pd
from .command_ids import GasSensorCommands, SGasConcentration1

class HIDGasSensor:
    VENDOR_ID = 0x1FC9  # NXP Vendor ID
    PRODUCT_ID = 0x8251  # Celisca Product ID
    SERIAL_NUMBER = "0024002KMV722ETRKK"  # Your sensor serial number
    WAIT_INTERVAL = 1  # ⏳ 1 second delay between readings
    CSV_FILE = "gas_sensor_readings.csv"  # 📂 CSV file to store readings

    # VOC Calibration Factor (Adjusted based on observed differences)
    VOC_CALIBRATION_FACTOR = 100  # Adjust this based on testing

    def __init__(self):
        self.device = self._connect_device()
        self._initialize_csv()

    def _connect_device(self):
        """Connects to the HID gas sensor using its serial number."""
        try:
            dev = hid.device()
            dev.open(self.VENDOR_ID, self.PRODUCT_ID, serial_number=self.SERIAL_NUMBER)
            dev.set_nonblocking(1)
            print(f"✅ Connected to sensor with Serial Number: {self.SERIAL_NUMBER}")
            return dev
        except Exception as e:
            print(f"❌ Error connecting to device: {e}")
            return None

    def _initialize_csv(self):
        """Creates the CSV file if it doesn't exist and writes headers."""
        try:
            with open(self.CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:  # If file is empty, write header
                    writer.writerow(["Timestamp", "Temperature (°C)", "VOC (Index)"])
        except Exception as e:
            print(f"⚠️ Error initializing CSV file: {e}")

    def send_command(self, command_id):
        """Sends a command to the device and reads the response."""
        if not self.device:
            print("❌ Device not connected")
            return None
        
        command = [0x00, command_id] + [0] * (64 - 2)  # 64-byte buffer
        self.device.write(command)
        time.sleep(0.1)  # Allow time for response

        data = self.device.read(64)  # Read full 64-byte response
        if not data:
            print("❌ No data received")
            return None

        return data

    def read_gas_data(self):
        """Reads gas concentration (VOC & Temperature) from the sensor."""
        raw_data = self.send_command(GasSensorCommands.CommandGetGasConcentration1.value)
        if not raw_data:
            return None

        try:
            gas_data = SGasConcentration1.from_bytes(raw_data)
            return gas_data
        except ValueError as e:
            print(f"❌ Error processing gas data: {e}")
            return None

    def log_to_csv(self, temperature, voc):
        """Saves the gas readings with timestamp to a CSV file."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, round(temperature, 2), round(voc, 1)])  # Adjusted for better accuracy
        except Exception as e:
            print(f"⚠️ Error writing to CSV: {e}")

    def close(self):
        """Closes the HID connection."""
        if self.device:
            self.device.close()
            print("🔌 Connection closed")

# 🚀 Continuous Reading Loop (Every 1s)
if __name__ == "__main__":
    sensor = HIDGasSensor()
    readings = []  # Store readings for interpolation
    try:
        while True:
            gas_data = sensor.read_gas_data()
            if gas_data:
                temperature = gas_data.temperature_c
                voc = gas_data.tvoc_ppm * HIDGasSensor.VOC_CALIBRATION_FACTOR  # Apply correction factor

                print(f"🌡️ Temperature: {temperature:.2f}°C | 🏭 VOC: {voc:.1f} Index")

                # Store readings for interpolation
                readings.append({"Timestamp": datetime.now(), "Temperature": temperature, "VOC": voc})

                # Save to CSV
                sensor.log_to_csv(temperature, voc)

            time.sleep(HIDGasSensor.WAIT_INTERVAL)  # ⏳ Wait 1 second

    except KeyboardInterrupt:
        print("\n🛑 Stopping VOC & Temperature Readings...")

        # Convert readings to DataFrame and interpolate missing values
        df = pd.DataFrame(readings)
        df.set_index("Timestamp", inplace=True)
        df = df.resample("1s").interpolate()  # Linear interpolation every 1 second

        # Save the interpolated data back to CSV
        df.to_csv("gas_sensor_readings_interpolated.csv")
        print("📂 Interpolated data saved: gas_sensor_readings_interpolated.csv")

    finally:
        sensor.close()
