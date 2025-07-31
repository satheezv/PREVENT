import py_trees
import numpy as np
import time
import cv2
from collections import Counter
from utils.Interface_ViTL14_FT import CLIPUtility
from utils.realSenseCamera import RealSenseCamera




class CLIPImageTextMatching(py_trees.behaviour.Behaviour):
    """
    A PyTrees behavior that performs image-text matching using CLIP ViT-L/14.
    """

    def __init__(self, name="CLIPImageTextMatching", text_labels=[], camera=None):
        super(CLIPImageTextMatching, self).__init__(name)
        self.camera = camera
        self.object_labels = text_labels
        self.matching_scores = []
        self.clip_util = CLIPUtility()  # Load the utility class

        # Blackboard for sharing data
        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="Emergency_stop_triggered", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="identified_hazard_infront", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="hazard_pic", access=py_trees.common.Access.WRITE)  

    def initialise(self):
        """Starts the camera before capturing."""
        if self.camera and not self.camera.is_running:
            self.camera.start_camera()

    def update(self):
        """Performs image-text matching and returns the best match."""
        try:
            if self._blackboard.Emergency_stop_triggered:
                timestamp = time.strftime("%H%M%S")
                filename = f"realsense_{timestamp}.png"

                # 🔹 Step 1: Capture the image
                self.image = self.camera.capture_image()
                self._blackboard.hazard_pic = self.image

                # Save the image
                cv2.imwrite(filename, self.image)
                print(f"Image saved as {filename}")
                # 🔹 Step 2: Identify the object
                best_object_match, object_score = self.clip_util.match_image_to_text(self.image, self.object_labels)

                if best_object_match:
                    # Store results in blackboard
                    self._blackboard.identified_hazard_infront = best_object_match

                    # print(f"[{self.name}] 🔍 Identified Object: {best_object_match[0]} (Confidence: {best_object_match[1]:.2f})")

                    print(f"[{self.name}] 🔍 Identified Object: {best_object_match} (Confidence: {object_score:.2f})")

                    if object_score:
                        print(f"[{self.name}] 📊 List of object scores: {object_score}")
                else:
                    print(f"[{self.name}] ❌ Could not recognize object.")
                    self._blackboard.identified_hazard_infront = None
                return py_trees.common.Status.SUCCESS
                
            else:
                print(f"[{self.name}] ✅ No obstacles detected.")
                return py_trees.common.Status.RUNNING

        except Exception as e:
            print(f"[{self.name}] ❌ Error: {e}")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Cleanup if needed."""
        print(f"[{self.name}] 🛑 Behavior Terminated.")
