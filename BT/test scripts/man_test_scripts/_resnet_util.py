import rospy
from kmriiwa_chemist_msgs.msg import RobotCommand, LBRCommand, KMPCommand, TaskStatus
import pyrealsense2 as rs
import cv2
import os
import numpy as np
import datetime
import time
import signal
import sys  # Required for sys.exit()
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image

class kukaRobotStatus:
    def __init__(self, topic_name, msg_type):
        """
        Utility class to subscribe to a ROS topic and store the latest message.
        :param topic_name: Name of the ROS topic
        :param target_node_id: Target node ID to check if the robot has reached the destination
        """
        self.topic_name = topic_name
        self.msg_type = msg_type
        # self.target_node_id = target_node_id
        self.latest_message = None

        # Initialize ROS node if it hasn't been initialized
        if not rospy.core.is_initialized():
            rospy.init_node("ros_topic_reader", anonymous=True)

        self.subscriber = rospy.Subscriber(self.topic_name, msg_type, self.callback)

    def callback(self, msg):
        """Callback function to store the latest received RobotStatus message."""
        self.latest_message = msg

    def get_latest_message(self):
        """Returns the latest received RobotStatus message."""
        return self.latest_message

# Gracefully handle Ctrl+C
def signal_handler(sig, frame):
    print("KeyboardInterrupt detected. Shutting down safely...")
    rospy.signal_shutdown("KeyboardInterrupt")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_last_image_index(save_path):
    """Scans folder and returns the last image index for sequential naming."""
    existing_files = [f for f in os.listdir(save_path) if f.startswith("image_") and f.endswith(".jpg")]
    existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
    return max(existing_indices) + 1 if existing_indices else 0

def publish_ros_msg(program_name, seq_id, task_type, node_id):
    pub = rospy.Publisher("/kuka2/commands", RobotCommand, queue_size=10)
    rospy.sleep(2)  # Delay for publisher readiness

    kuka_cmd = RobotCommand()
    kuka_cmd.seq = seq_id
    kuka_cmd.priority_task = False
    kuka_cmd.task_type = task_type
    kuka_cmd.task_name = "Data Collection"

    arm_command = LBRCommand()
    base_command = KMPCommand()

    base_command.graph_id = 1
    base_command.node_id = node_id  # Target node
    base_command.fine_localization = True

    arm_command.program_name = program_name
    arm_command.program_parameters = []

    kuka_cmd.arm_command = arm_command
    kuka_cmd.base_command = base_command

    pub.publish(kuka_cmd)
    _time = datetime.datetime.now()
    print(f"Published command: {program_name} at {_time}")

def capture_image(image_count):
    save_path = "Dataset_chemspeed\\combined_test_13_06_25"
    os.makedirs(save_path, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
    pipeline.start(config)

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if color_frame:
        image = np.asanyarray(color_frame.get_data())
        image_index = get_last_image_index(save_path)
        image_name = os.path.join(save_path, f"image_{image_index:04d}.jpg")
        cv2.imwrite(image_name, image)
        print(f"Image saved: {image_name}")

    pipeline.stop()
    return image_name

def test_resnet18(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Two classes: Empty Floor vs Obstruction
    model.load_state_dict(torch.load("ztests\man_test_scripts\\resnet18_chemspeed_plate_detection_40_v2.pth", map_location=device))
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        class_names = ['hazard', 'no hazard']

    print(f"ResNet-18 Prediction for {image_path}: {class_names[predicted.item()]}")

    return class_names[predicted.item()]

def is_kuka_finished(ros_topic_reader,seq, mode):
    msg:TaskStatus = ros_topic_reader.get_latest_message()
    if msg is not None:
        if msg.cmd_seq == seq:
            if mode == "LBR":
                if msg.task_lbr_state == TaskStatus.FINISHED:
                    return True
                else:
                    return False
            else:
                if msg.task_kmp_state == TaskStatus.FINISHED:
                    return True
                else:
                    return False
        else:
            return False
    else:
        print("Msg not received yet")
        return False





def write_results_to_file(results, file_path="results.txt"):
    """Write prediction results to a text file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("Iteration Results\n")
        file.write("==================\n")
        for i, prediction in enumerate(results, start=1):
            file.write(f"Iteration {i}: Prediction - {prediction}\n")
        file.write("==================\n")
        file.write("Results successfully saved.\n")

def main():
    rospy.init_node("kuka_Data_collection_node", anonymous=True)
    ros_topic_reader = kukaRobotStatus("/kuka2/task_status", msg_type = TaskStatus)
    rospy.sleep(3)
    _seq = 1
    result = []  # Store predictions
    # publish_ros_msg("CS_CalibrationAndMoveCheck", _seq, 2, 27)  # Home message
    # while not is_kuka_finished(ros_topic_reader,_seq, mode="LBR"):  # Loop until the robot finishes the task
    #     time.sleep(1)  # Check the status every second
    for i in range(5):
        print(f"Iteration {i+1} of 5")

    #     publish_ros_msg("CS_MoveCheck", _seq, 0, 0)  # Task message
    #     time.sleep(4)  # Wait for robot to reach the target
    #     _seq += 1

    #     image_path = capture_image(i + 1)  # Capture image after task completion
    #     prediction = test_resnet18(image_path)

    #     result.append(prediction)

    #     publish_ros_msg("CS_MoveApproach", _seq, 0, 0)  # Home message
    #     time.sleep(4)  # Wait for robot to return home
    #     _seq += 1
    #     print("last seq id: ", _seq)


    # # Write results to file
    #     write_results_to_file(result, "robot_predictions_test.txt")
    #     print("Results saved to 'robot_predictions_test.txt'")

        publish_ros_msg("CS_CalibrationAndRackTransfer", _seq, 0, 0)  # Task message
        time.sleep(4)  # Wait for robot to reach the target
        _seq += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program stopped by user.")
