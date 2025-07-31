
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Define the coordinate you want to get depth for (e.g., center)
        x, y = 320, 240  # Change to your coordinate

        # Get depth in meters
        depth_value = depth_frame.get_distance(x, y)

        # Display
        print(f"Depth at ({x}, {y}) = {depth_value:.3f} meters")

        # Optional: Display the image
        cv2.imshow('Depth Frame', depth_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
