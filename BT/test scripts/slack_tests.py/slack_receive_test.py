import time
from slack_sdk import WebClient

SLACK_BOT_TOKEN = "xoxb-638030615362-8306647721012-l6SPuh6GTe8bTaIjMCNEEbS3"
USER_ID = "your-user-id-here"

client = WebClient(token=SLACK_BOT_TOKEN)

while True:
    response = client.conversations_history(channel=USER_ID, limit=1)
    if response["messages"]:
        latest_message = response["messages"][0]["text"]
        print("Latest Message:", latest_message)

    time.sleep(5)  # Check for new messages every 5 seconds