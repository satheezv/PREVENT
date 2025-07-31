import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import signal
from utils.kuka_send_nav_msg import KukaTaskPublisher
from utils.kuka_get_status import kukaRobotStatus

class RealSenseCamera:
    """
    Utility to continuously capture RGB images from an Intel RealSense camera
    and save them with iterating filenames until manually stopped.
    """

    def __init__(self, save_folder=r"W:\\gasSensor_ws\\others\\data v3\\fps6\\object_on_the_floor", width=640, height=480, fps=6):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.is_running = False
        self.save_folder = save_folder
        self.running = True  # Track if capturing should continue

        # Configure the RealSense camera (RGB only)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        # Create save directory if it doesn't exist
        os.makedirs(self.save_folder, exist_ok=True)

        # Handle Ctrl+C to stop gracefully
        signal.signal(signal.SIGINT, self.stop_on_signal)

    def start_camera(self):
        """Starts the RealSense camera."""
        if not self.is_running:
            self.pipeline.start(self.config)
            self.is_running = True
            print("✅ Camera started!")

    def get_next_filename(self):
        """
        Scans the folder and finds the next available filename.
        Ensures numbering continues correctly.
        """
        existing_files = [f for f in os.listdir(self.save_folder) if f.startswith("image_") and f.endswith(".jpg")]
        existing_numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
        next_number = max(existing_numbers) + 1 if existing_numbers else 0
        return os.path.join(self.save_folder, f"image_{next_number:04d}.jpg")

    def capture_image(self):
        """Captures a single RGB image from the RealSense camera."""
        try:
            if not self.is_running:
                print("❌ Cannot capture image, camera is not started!")
                return None

            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("⚠️ No color frame detected!")
                return None

            return np.asanyarray(color_frame.get_data())

        except Exception as e:
            print(f"❌ Error capturing image: {e}")
            return None

    def stop_camera(self):
        """Stops the RealSense camera."""
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
            print("🛑 Camera stopped.")
        else:
            print("⚠️ Camera was not running, skipping stop.")

    def stop_on_signal(self, signum, frame):
        """Handles Ctrl+C to stop the loop."""
        print("\n🛑 Ctrl+C detected! Stopping image capture...")
        self.running = False  # Stops the while loop

# Main Execution
if __name__ == "__main__":
    # Initialize camera and robot communication
    camera = RealSenseCamera()
    target_node_id = 24
    seq_id = 6
    kuka = KukaTaskPublisher(node_id=target_node_id, task_name="navigation_data_collection", task_type=1, seq_id=seq_id)
    kuka_status = kukaRobotStatus(topic_name="/kuka2/robot_status")

    # Publish the task and start the camera
    camera.start_camera()
    kuka.publish_task()
    

    try:
        while True:
            # Capture and save image
            image = camera.capture_image()
            if image is not None:
                filename = camera.get_next_filename()
                cv2.imwrite(filename, image)
                print(f"✅ Saved: {filename}")

            # Check robot status
            received_message = kuka_status.get_latest_message()
            if received_message and received_message.last_graph_node_id == target_node_id:
                print("🎯 Target node reached. Stopping camera.")
                break

            time.sleep(1)  # Prevent excessive polling

    finally:
        camera.stop_camera()
        print("✅ Camera stopped.")
