import rospy
from kmriiwa_chemist_msgs.msg import RobotCommand, LBRCommand, KMPCommand
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

# Gracefully handle Ctrl+C
def signal_handler(sig, frame):
    print("\n🛑 KeyboardInterrupt detected. Shutting down safely...")
    rospy.signal_shutdown("KeyboardInterrupt")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_last_image_index(save_path):
    """Scans folder and returns the last image index for sequential naming."""
    existing_files = [f for f in os.listdir(save_path) if f.startswith("image_") and f.endswith(".jpg")]
    existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
    return max(existing_indices) + 1 if existing_indices else 0

def publish_ros_msg(program_name, seq_id, task_type):
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
    base_command.node_id = 30  # Target node
    base_command.fine_localization = True

    arm_command.program_name = program_name
    arm_command.program_parameters = []

    kuka_cmd.arm_command = arm_command
    kuka_cmd.base_command = base_command

    pub.publish(kuka_cmd)
    _time = datetime.datetime.now()
    print(f"Published command: {program_name} at {_time}")

def capture_image(image_count):
    save_path = "W:\gasSensor_ws\others\data for resnet\man_dataset\\test\exp2 - no hazard 25 runs 18_3_25"
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
        print(f"✅ Image saved: {image_name}")

    pipeline.stop()
    return image_name

def test_resnet18(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Two classes: Empty Floor vs Obstruction
    model.load_state_dict(torch.load("W:\gasSensor_ws\GasSensor_ws\src\BT\ztests\man_test_scripts\\resnet18_rack_detection_v3.pth", map_location=device))
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

    print(f"🔍 ResNet-18 Prediction for {image_path}: {class_names[predicted.item()]}")

    return class_names[predicted.item()]

def write_results_to_file(results, file_path="results.txt"):
    """Write prediction results to a text file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("Iteration Results\n")
        file.write("==================\n")
        for i, prediction in enumerate(results, start=1):
            file.write(f"Iteration {i}: Prediction - {prediction}\n")
        file.write("==================\n")
        file.write("✅ Results successfully saved.\n")



def main():
    rospy.init_node("kuka_task_pub_node", anonymous=True)
    _seq = 5
    result = []  # Store predictions

    for i in range(25):
        print(f"Iteration {i+1} of 25")
        publish_ros_msg("MoveCheck", _seq, 0)  # Task message
        time.sleep(7)  # Wait for robot to reach the target
        _seq += 1

        image_path = capture_image(i + 1)  # Capture image after task completion
        prediction = test_resnet18(image_path)

        result.append(prediction)

        publish_ros_msg("MoveApproach", _seq, 0)  # Home message
        time.sleep(7)  # Wait for robot to return home
        _seq += 1
        print("last seq id: ", _seq)

    # Write results to file
        write_results_to_file(result, "robot_predictions.txt")
        print("Results saved to 'robot_predictions.txt'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program stopped by user.")
