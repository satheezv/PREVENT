import py_trees
import numpy as np
import time
import cv2
from utils.Interface_ViTL14 import CLIPUtility
from utils.realSenseCamera import RealSenseCamera




class CLIPImageTextMatching(py_trees.behaviour.Behaviour):
    """
    A PyTrees behavior that performs image-text matching using CLIP ViT-L/14.
    """

    def __init__(self, name="CLIPImageTextMatching", text_labels=[]):
        super(CLIPImageTextMatching, self).__init__(name)
        self.camera = RealSenseCamera()


        self.confirmation_labels = ["clean empty floor", "empty floor with a line", "object on the floor", "spillage on the floor"]
        self.object_labels = text_labels
        self.matching_scores = []
        self.clip_util = CLIPUtility()  # Load the utility class

        # Blackboard for sharing data
        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(key="is_obstacle_detected", access=py_trees.common.Access.WRITE)
        self._blackboard.register_key(key="clip_best_match", access=py_trees.common.Access.WRITE)
        self._blackboard.register_key(key="clip_scores", access=py_trees.common.Access.WRITE)

    def initialise(self):
        """Starts the camera before capturing."""
        self.camera.start_camera()
        self._blackboard.is_obstacle_detected = False

    def update(self):
        """Performs image-text matching and returns the best match."""
        try:
            # 🔹 Get the latest image from the blackboard
            if not self.camera.is_running:
                self.camera.start_camera()
                      
            self.image = self.camera.capture_image()

            # 🔹 Validate image before processing
            if self.image is None or not isinstance(self.image, np.ndarray):
                print(f"[{self.name}] ❌ No valid image found in blackboard!")
                return py_trees.common.Status.FAILURE

            if not self.object_labels:
                print(f"[{self.name}] ❌ No text labels provided!")
                return py_trees.common.Status.FAILURE

            # 🔹 Step 1: Check if there's an object on the floor
            best_match, confirmation_scores = self.clip_util.match_image_to_text(self.image, self.confirmation_labels)

            if best_match is None:
                print(f"[{self.name}] ❌ No valid match found.")
                return py_trees.common.Status.FAILURE

            print(f"[{self.name}] 🔍 Best Match: {best_match[0]} (Confidence: {best_match[1]:.2f})")

            if best_match[0] == "object on the floor" or best_match[0] == "spillage on the floor":
                print(f"[{self.name}] ⚠️ Obstacle detected on the floor!")
                self._blackboard.is_obstacle_detected = True
                timestamp = time.strftime("%H%M%S")
                filename = f"realsense_{timestamp}.png"

                # Save the image
                cv2.imwrite(filename, self.image)
                print(f"Image saved as {filename}")
                # 🔹 Step 2: Identify the object
                best_object_match, object_scores = self.clip_util.match_image_to_text(self.image, self.object_labels)

                if best_object_match:
                    # Store results in blackboard
                    self._blackboard.clip_best_match = best_object_match[0]
                    self._blackboard.clip_scores = object_scores

                    print(f"[{self.name}] 🔍 Identified Object: {best_object_match[0]} (Confidence: {best_object_match[1]:.2f})")

                    if object_scores:
                        print(f"[{self.name}] 📊 List of object scores: {object_scores}")

                    return py_trees.common.Status.SUCCESS
                else:
                    print(f"[{self.name}] ❌ Could not recognize object.")
                    return py_trees.common.Status.FAILURE
            else:
                print(f"[{self.name}] ✅ No obstacles detected.")
                return py_trees.common.Status.RUNNING

        except Exception as e:
            print(f"[{self.name}] ❌ Error: {e}")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Cleanup if needed."""
        print(f"[{self.name}] 🛑 Behavior Terminated.")
