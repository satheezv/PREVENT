import pyrealsense2 as rs
import numpy as np
import cv2
import time

class RealSenseCamera:
    """
    Utility to capture images from an Intel RealSense camera.
    """

    def __init__(self, width=640, height=480, fps=15):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.is_running = False  # Track whether the camera is running

        # Configure the RealSense camera
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

    def start_camera(self):
        """
        Starts the RealSense camera.
        """
        if not self.is_running:  # Prevent multiple starts
            self.pipeline.start(self.config)
            self.is_running = True

    def capture_image(self):
        """
        Captures a single image from the RealSense camera.

        :return: The captured image as a NumPy array.
        """
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
            image = np.asanyarray(color_frame.get_data())
            return image  # Return image instead of saving

        except Exception as e:
            print(f"❌ Error capturing image: {e}")
            return None



    def is_blurry(self, image, threshold=80):
        """
        Detects if an image is blurry using the Variance of Laplacian method.
        Prints the Laplacian variance for debugging.
        
        :param image: Input image as a NumPy array (BGR or Grayscale).
        :param threshold: Threshold value for blurriness detection (lower = more blurry).
        :return: True if blurry, False otherwise.
        """
        if len(image.shape) == 3:  # Convert BGR to Grayscale if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        variance = cv2.Laplacian(image, cv2.CV_64F).var()
        
        # print(f"📸 Frame Blurriness Score: {variance:.2f}")  # Print blurriness score
        
        return variance < threshold  # Returns True if the image is blurr

    def capture_images(self, duration=1):
        """
        Captures multiple images from the RealSense camera while **ignoring blurry frames**.

        :param duration: Total time (in seconds) for which images are captured.
        :return: A list of sharp images as NumPy arrays.
        """
        try:
            if not self.is_running:
                print("❌ Cannot capture images, camera is not started!")
                return []

            num_frames = self.fps * duration  # Total frames to capture
            captured_images = []

            for frame_index in range(num_frames):
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    print("❌ No color frame detected! Skipping frame.")
                    continue

                # Extract image from frame
                image = np.asanyarray(color_frame.get_data())

                # Check if the image is blurry
                if self.is_blurry(image, threshold=20):  # Adjust threshold as needed
                    print(f"⚠️ Frame {frame_index} is blurry. Ignoring.")
                    continue  # Skip blurry frames

                # Append only if the frame index is odd (original condition)
                if frame_index % 3 == 0:
                    captured_images.append(image)
                    print(f"✅ Frame {frame_index} captured successfully.")

                # Sleep to maintain the desired FPS
                time.sleep(1 / self.fps)

            return captured_images
        
        except Exception as e:
            print(f"❌ Error capturing images: {e}")
            return []


    def stop_camera(self):
        """
        Stops the RealSense camera.
        """
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False  # Mark camera as stopped
        else:
            print("⚠️ Camera was not running, skipping stop.")
