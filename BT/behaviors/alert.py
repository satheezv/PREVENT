import py_trees
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class Alert(py_trees.behaviour.Behaviour):
    def __init__(self, name, slack_token, slack_user_id, msg_to_send):
        super(Alert, self).__init__(name)
        self.slack_client = WebClient(token=slack_token)
        self.slack_user_id = slack_user_id  # Use Slack user ID for self-messaging
        self.msg = msg_to_send

    # def setup(self, timeout):
    #     # Perform any initialization required, e.g., setting up Slack client
    #     print(f"Setting up Alert with timeout: {timeout}")
    #     return True

    def initialise(self):
        # Called just before entering RUNNING state
        pass

    def update(self):
        # Action to perform during execution
        try:
            alert_message = f"Hazard detected! {self.msg}"
            print(alert_message)
            # Send a self-message via Slack
            self.send_slack_message(alert_message)
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"Failed to send alert: {e}")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        # Clean up when exiting the behaviour
        pass

    def send_slack_message(self, message):
        try:
            response = self.slack_client.chat_postMessage(
                channel=self.slack_user_id,  # Send message to self
                text=message
            )
            print(f"Slack message sent: {response['message']['text']}")
        except SlackApiError as e:
            print(f"Error sending message to Slack: {e.response['error']}")
