from flask import Flask, request, jsonify
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import json
import io
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import time

# # Initialize Flask app
# app = Flask(__name__)

class SlackAlertUtility:
    def __init__(self, user_name):
        """
        Utility class to send Slack alerts.
        :param slack_token: Slack API token
        :param slack_user_id: Slack User ID for direct messaging
        """
        
        self.json_file_path = r"W:\\gasSensor_ws\\GasSensor_ws\src\BT\\utils\slack_user_id.json"
        self.SLACK_TOKEN = "xoxb-638030615362-8306647721012-l6SPuh6GTe8bTaIjMCNEEbS3"
        self.slack_client = WebClient(token=self.SLACK_TOKEN)
        self.username = user_name
        self.slack_user_id = self.get_user_id_from_json()
    
    def send_slack_message(self, message):
        """Send a Slack message to the user."""
        try:
            response = self.slack_client.chat_postMessage(
                channel=self.slack_user_id,
                text=message
            )
            print(f"Slack message sent: {response['message']['text']}")
            return True
        except SlackApiError as e:
            print(f"Error sending message to Slack: {e.response['error']}")
            return False
        
    def send_slack_message_with_csv(self, file_path, filename="data.csv", title="Here is your data file!"):
        """Send a Slack message with a CSV file attachment."""
        try:
            with open(file_path, 'rb') as file_data:
                response = self.slack_client.files_upload_v2(
                    channel=self.get_dm_channel(self.slack_user_id),
                    file=file_data,
                    filename=filename,
                    title=title
                )

            if response["ok"]:
                print("✅ CSV file uploaded successfully!")
                print("📎 File URL:", response["file"]["permalink"])
            else:
                print(f"❌ Error uploading file: {response['error']}")

        except SlackApiError as e:
            print(f"❌ Error uploading file: {e.response['error']}")
        except FileNotFoundError:
            print(f"❌ Error: File '{file_path}' not found.")

    def send_slack_message_with_image(self, image):
        """Send a Slack message with an attachment to the user."""
        try:
            if isinstance(image, np.ndarray):  # Check if image is from OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                image = Image.fromarray(image)  # Convert NumPy array to PIL image
    
            _image_data = io.BytesIO()
            image.save(_image_data, format="PNG")  # Ensure it's saved in a compatible format
            _image_data.seek(0)  # Move pointer to the start
            response = self.slack_client.files_upload_v2(
channel=self.get_dm_channel(self.slack_user_id),  # Note: `channel_id` instead of `channels`
file=_image_data,
filename="Hazard.jpg",
title="Here is an image from Kuka 2!"
)
            if response["ok"]:
                print("File uploaded successfully!")
                print("File URL:", response["file"]["permalink"])
            else:
                print(f"Error uploading file: {response['error']}")
        except SlackApiError as e:
            print(f"Error uploading file: {e.response['error']}")
    
    def get_dm_channel(self, user_id):
        response = self.slack_client.conversations_open(users=[user_id])
        return response["channel"]["id"]
    
    def read_latest_message(self):
        """Read the latest message from the user."""
        try:
            response = self.slack_client.conversations_history(
                channel=self.get_dm_channel(self.slack_user_id),  # Read messages from self
                limit=1
            )
            if response["messages"]:
                latest_message = response["messages"][0]["text"]
                print(f"Latest message: {latest_message}")
                return latest_message
            else:
                print("No messages found.")
                return None
        except SlackApiError as e:
            print(f"Error reading messages from Slack: {e.response['error']}")
            return
        
    def download_csv_from_slack(self):
        """Download the latest CSV file uploaded by either the bot or the user."""
        try:
            # Fetch recent messages to identify files from both sides
            messages = self.slack_client.conversations_history(
                channel=self.get_dm_channel(self.slack_user_id),
                limit=10  # Increase limit if needed
            )

            for message in messages['messages']:
                if 'files' in message:
                    for file in message['files']:
                        if file['name'].endswith('.csv'):
                            download_url = file['url_private']

                            # Download the file
                            headers = {'Authorization': f'Bearer {self.SLACK_TOKEN}'}
                            file_response = requests.get(download_url, headers=headers)

                            # Save file locally
                            file_path = f"W:\gasSensor_ws\GasSensor_ws\src\BT\ztests\slack_tests.py\{file['name']}"
                            with open(file_path, 'wb') as file:
                                file.write(file_response.content)

                            print(f"✅ CSV file downloaded successfully: {file_path}")

                            # Load and display content
                            data = pd.read_csv(file_path)
                            print("🔍 CSV Data Preview:")
                            print(data.head())
                            return data
            
            print("❌ No CSV file found in recent Slack uploads.")
        except SlackApiError as e:
            print(f"❌ Slack API Error: {e.response['error']}")
        except Exception as e:
            print(f"❌ Error downloading CSV file: {e}")

        
    def get_user_id_from_json(self):
        """
        Reads a JSON file and retrieves the Slack user ID for the given username.
        :return: Slack user ID if found, else None
        """
        try:
            with open(self.json_file_path, 'r') as file:
                data = json.load(file)

            for user in data.get("members", []):  # Assuming JSON contains a "members" list
                if user.get("name") == self.username:
                    return user.get("id")  # Returning Slack User ID

            print(f"Username '{self.username}' not found in JSON.")
            return None
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading JSON file: {e}")
            return None



if __name__ == "__main__":
    slack = SlackAlertUtility(user_name="Satheesh")
    slack.send_slack_message(message="Hi, This is a test msg from OT2 slack bot :slightly_smiling_face:")
    slack.send_slack_message_with_csv(file_path="W:\gasSensor_ws\GasSensor_ws\src\BT\gas_sensor_readings_interpolated.csv", filename="Optimised_values.csv")
    time.sleep(5)
    slack.download_csv_from_slack()
