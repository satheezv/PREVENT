import py_trees
from utils.realSenseCamera import RealSenseCamera

class CaptureImageBehavior(py_trees.behaviour.Behaviour):
    """
    A PyTrees behavior that captures an image from a RealSense camera
    and stores it in the blackboard.
    """

    def __init__(self, name="CaptureImageBehavior"):
        super(CaptureImageBehavior, self).__init__(name)
        self.camera = RealSenseCamera()
        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(key="captured_image", access=py_trees.common.Access.WRITE)

    def initialise(self):
        """Starts the camera before capturing."""
        self.camera.start_camera()

    def update(self):
        """Captures an image and stores it in the blackboard."""
        if not self.camera.is_running:
            self.camera.start_camera()
                      
        image = self.camera.capture_image()

        if image is not None:
            self._blackboard.captured_image = image  # Store in blackboard
            print(f"[{self.name}] 📸 Image captured and stored in blackboard.")
            return py_trees.common.Status.SUCCESS
        else:
            print(f"[{self.name}] ❌ Failed to capture image.")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Stops the camera after capturing the image."""
        if self.camera.is_running:  # Ensure camera was started before stopping
            self.camera.stop_camera()
            print(f"[{self.name}] 🛑 Camera stopped.")
        else:
            print(f"[{self.name}] ⚠️ Camera was not started, skipping stop.")