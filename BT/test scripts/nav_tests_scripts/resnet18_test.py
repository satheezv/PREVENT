import py_trees
import time
import cv2
import torch
from utils.realSenseCamera import RealSenseCamera
from behaviors.sensor.GetVisionPrediction_ResNet18 import ResNetBehavior

# Configuration
MODEL_PATH = "W:\gasSensor_ws\GasSensor_ws\src\BT\ztests\\resnet18_floor_detection.pth"  # Path to trained ResNet model
CLASS_MAP = {"0": "empty_floor", "1": "object_on_floor"}  # Class mapping
FPS = 6  # Frames per second

# Initialize RealSense Camera
camera = RealSenseCamera()
camera.start_camera()

# Initialize Behavior
behavior = ResNetBehavior(name="TestResNetBehavior", model_path=MODEL_PATH, camera=camera)

# Create a PyTrees root behavior tree
root = py_trees.composites.Sequence("Root", memory=True)
root.add_child(behavior)

# Setup the behavior tree
behavior_tree = py_trees.trees.BehaviourTree(root)

try:
    print("[INFO] Starting behavior test. Press CTRL+C to stop.")
    while True:
        start_time = time.time()
        
        # Tick behavior tree
        behavior_tree.tick()
        
        # Display latest captured frame (optional)
        images = camera.capture_images(duration=1)
        if images:
            frame = images[0]
            cv2.imshow("RealSense Frame", frame)
            
        # Limit FPS
        elapsed_time = time.time() - start_time
        time.sleep(max(1/FPS - elapsed_time, 0))
        
        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] Stopping behavior test.")
finally:
    # Cleanup
    camera.stop_camera()
    cv2.destroyAllWindows()
