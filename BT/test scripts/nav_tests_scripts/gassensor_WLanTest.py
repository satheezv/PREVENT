import requests
import json
import time
import csv
import pandas as pd
from datetime import datetime
from .command_ids import GasSensorCommands

class NetworkGasSensor:
    SENSOR_IP = "192.168.137.1"
    SENSOR_PORT = 9020
    WAIT_INTERVAL = 1  # ⏳ 1 second delay between readings
    CSV_FILE = "network_gas_sensor_readings.csv"  # 📂 CSV file to store readings
    JSON_FILE = "Gas_Sensor.json"  # JSON file to store raw data

    def __init__(self):
        self.endpoint = f"http://{self.SENSOR_IP}:{self.SENSOR_PORT}/celiscaIOTMiddleware/"
        self.activated = False
        self.readings = []  # Store readings for interpolation
        self._initialize_csv()

    def _initialize_csv(self):
        """Creates the CSV file if it doesn't exist and writes headers."""
        try:
            with open(self.CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:  # If file is empty, write header
                    writer.writerow(["Timestamp", "Temperature (°C)", "VOC (Index)"])
        except Exception as e:
            print(f"⚠️ Error initializing CSV file: {e}")

    def request_data(self, stop=False):
        """Send a request to the sensor to fetch data."""
        payload = {
            "Command": GasSensorCommands.CommandGetGasConcentration1.value,
            "Stop": stop,
            "Timestamp": datetime.now().isoformat()
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Request failed with status {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"❌ Error requesting data: {e}")
            return None

    def activate(self):
        """Activate the sensor to start periodic data fetching."""
        self.activated = True
        print(f"✅ Sensor activated with {self.WAIT_INTERVAL} sec interval")
        while self.activated:
            data = self.request_data()
            if data:
                readings = self.parse_json_data(data)
                for reading in readings:
                    print(f"📡 {reading['Timestamp']} | Temp: {reading['Temperature']}°C | VOC: {reading['VOC']} Index")
                    self.log_to_csv(reading)
                    self.readings.append(reading)
            time.sleep(self.WAIT_INTERVAL)

    def deactivate(self):
        """Deactivate the sensor to stop data fetching."""
        self.activated = False
        print("🛑 Sensor deactivated")
        self.save_interpolated_csv()

    def parse_json_data(self, json_data):
        """Extracts temperature and VOC readings from sensor JSON data."""
        readings = []
        for dataset in json_data.get("DataSetList", []):
            timestamp = dataset.get("DateTime", "Unknown Time")
            temp, voc = None, None
            for sensor in dataset.get("DataSet", []):
                if sensor.get("SensorName") == "BME68X":
                    for param in sensor.get("param", []):
                        if param["n"] == "Temperature":
                            temp = param["v"]
                        elif param["n"] == "Tvoc":
                            voc = param["v"]
            if temp is not None and voc is not None:
                readings.append({"Timestamp": timestamp, "Temperature": temp, "VOC": voc})
        return readings

    def log_to_csv(self, reading):
        """Saves the gas readings with timestamp to a CSV file."""
        try:
            with open(self.CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([reading["Timestamp"], round(reading["Temperature"], 2), round(reading["VOC"], 3)])
        except Exception as e:
            print(f"⚠️ Error writing to CSV: {e}")

    def save_interpolated_csv(self):
        """Interpolates missing values and saves final dataset."""
        df = pd.DataFrame(self.readings)
        df.set_index("Timestamp", inplace=True)
        df = df.resample("1S").interpolate()
        df.to_csv("network_gas_sensor_readings_interpolated.csv")
        print("📂 Interpolated data saved: network_gas_sensor_readings_interpolated.csv")

# Example Usage
if __name__ == "__main__":
    sensor = NetworkGasSensor()
    try:
        sensor.activate()
    except KeyboardInterrupt:
        sensor.deactivate()