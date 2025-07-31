from utils.slack_msging import SlackAlertUtility
import time


user_name = "Satheesh"

SlackInterface = SlackAlertUtility(user_name=user_name)


SlackInterface.send_slack_message("Hello, this is a test message from the test script.")
time.sleep(2)
SlackInterface.read_latest_message()
time.sleep(2)
SlackInterface.send_slack_message_with_image("Hello, this is a test message with attachment from the test script.", r"W:\\gasSensor_ws\\others\\nav experiment obs images\\exp_images_5_3_25\\resnet_191542.png")
time.sleep(2)
SlackInterface.read_latest_message()
time.sleep(2)
print("Test completed.")