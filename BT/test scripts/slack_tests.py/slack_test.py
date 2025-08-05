from slack_sdk import WebClient

SLACK_TOKEN = "xoxb-638030615362-8306647721012-l6SPuh6GTe8bTaIjMCNEEbS3"

client = WebClient(token=SLACK_TOKEN)
response = client.users_list()
print("users: ", response)

# for user in response['members']:
#     if not user['is_bot'] and user['name'] == "your-username":  # Replace "your-username" with your Slack username
#         print(f"Your Slack User ID is: {user['id']}")
