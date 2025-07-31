import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import signal

class RealSenseCamera:
    """
    Utility to continuously capture RGB images from an Intel RealSense camera
    and save them with iterating filenames until manually stopped.
    """

    def __init__(self, save_folder=r"W:\\gasSensor_ws\\others\data v3\\empty_floor_test", width=640, height=480, fps=6):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.is_running = False  # Track camera state
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

    def capture_and_save_images(self, interval=1):
        """
        Continuously captures and saves images every `interval` seconds.
        Runs indefinitely until manually stopped.
        """
        if not self.is_running:
            print("❌ Camera is not running! Starting camera now...")
            self.start_camera()

        print(f"📸 Starting continuous image capture... Saving to '{self.save_folder}' (Press Ctrl+C to stop)")

        while self.running:
            image = self.capture_image()
            if image is not None:
                filename = self.get_next_filename()
                cv2.imwrite(filename, image)
                print(f"✅ Saved: {filename}")
            else:
                print("⚠️ Skipping frame due to capture error.")

            time.sleep(interval)  # Wait for the next capture

        self.stop_camera()
        print("🛑 Image capture stopped.")

    def capture_image(self):
        """Captures a single RGB image from the RealSense camera."""
        try:
            if not self.is_running:
                print("❌ Cannot capture image, camera is not started!")
                return None

            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("❌ No color frame detected!")
                return None

            # Convert to NumPy array
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

# Run the image capture process
if __name__ == "__main__":
    camera = RealSenseCamera()
    camera.start_camera()
    camera.capture_and_save_images(interval=1)  # Capture image every second
