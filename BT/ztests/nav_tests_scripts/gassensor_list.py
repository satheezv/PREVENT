################## USB

# import hid

# def list_devices():
#     devices = hid.enumerate()
#     for dev in devices:
#         if dev['vendor_id'] == 0x1FC9 and dev['product_id'] == 0x8251:  # Change these if needed
#             print(f"Found Gas Sensor: Serial Number = {dev['serial_number']}")
            
# list_devices()



################### NETWORK

import requests
import json
import time
import csv
import pandas as pd
import socket
from datetime import datetime
from .command_ids import GasSensorCommands

class NetworkScanner:
    @staticmethod
    def scan_network(base_ip="192.168.137.", start=1, end=254, port=9020, timeout=0.5):
        """Scans the network to find the gas sensor's IP."""
        print("🔍 Scanning network for gas sensor...")
        found_ip = None
        for i in range(start, end + 1):
            ip = f"{base_ip}{i}"
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((ip, port))
                sock.close()
                if result == 0:
                    print(f"✅ Gas sensor detected at {ip}")
                    found_ip = ip
                    break
            except Exception as e:
                pass
        if not found_ip:
            print("❌ No gas sensor found on the network.")
        return found_ip

# Example usage
if __name__ == "__main__":
    sensor_ip = NetworkScanner.scan_network()
    if sensor_ip:
        print(f"🌐 Use this IP for sensor communication: {sensor_ip}")
