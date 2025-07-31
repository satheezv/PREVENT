import py_trees
import time
import rospy
import signal
from skills.SafeManipulation import create_manipulation_behavior_tree, signal_handler

class SafeManipulationTask:
    def __init__(
        self,
        destination_node_id = 27,
        user_name="Satheesh",
        kuka_sequence_id = 1,
        vision_model_path = "",
        vision_class_map = {},
    ):
        print("Starting Hazard Detection Behavior Tree... Press Ctrl+C to exit.")

        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="task_complete", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="kuka_seq_id", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(key="restart_tree", access=py_trees.common.Access.WRITE)

        self._blackboard.task_complete = False
        self._blackboard.kuka_seq_id = kuka_sequence_id
        self._blackboard.restart_tree = False

        self.destination_node_id = destination_node_id
        self.user_name = user_name
        self.vision_model_path = vision_model_path
        self.vision_class_map = vision_class_map

        self._safe_grasp_task_tree = create_manipulation_behavior_tree(task_node_id=self.destination_node_id, Resnet_vision_model_path=self.vision_model_path, vision_class_map=self.vision_class_map, user_name=self.user_name)
        signal.signal(signal.SIGINT, signal_handler)  # ✅ Capture Ctrl+C

        self._task_root = py_trees.composites.Sequence(name="task-preGraspInspection", memory=False)
        self._task_root.add_child(self._safe_grasp_task_tree)
        self._task_root.add_child(
            py_trees.behaviours.SetBlackboardVariable(
                name="task_complete",
                variable_name="task_complete",
                variable_value=True,
                overwrite=True,
            )
        )
        self._task_tree = py_trees.trees.BehaviourTree(self._task_root)

    def run(self):
        try:
            while (not self._blackboard.task_complete) and (not rospy.is_shutdown()):
                self._task_tree.tick()
                # print("Tick...")
                time.sleep(2)  # Tick the tree every 2 seconds

                # 🔹 Check if restart is needed
                if self._blackboard.restart_tree:
                    print("[INFO] Restarting Behavior Tree...")

                    # 🔹 Reset Blackboard Variables
                    self._blackboard.task_complete = False
                    self._blackboard.restart_tree = False  # Reset flag

                    # 🔹 Recreate the navigation subtree (clone a new instance)
                    _safe_grasp_task_tree = create_manipulation_behavior_tree(
                        self.destination_node_id, self.vision_model_path, self.vision_class_map, self.user_name
                    )

                    # 🔹 Recreate the behavior tree **without memory** (fresh start)
                    self._task_root = py_trees.composites.Sequence(name="task-preGraspInspection", memory=False)
                    self._task_root.add_child(_safe_grasp_task_tree)  # 🔹 Use a new instance
                    self._task_root.add_child(
                        py_trees.behaviours.SetBlackboardVariable(
                            name="task_complete",
                            variable_name="task_complete",
                            variable_value=True,
                            overwrite=True,
                        )
                    )

                    self._task_tree = py_trees.trees.BehaviourTree(self._task_root)  # Reinitialize the tree
                    self._task_tree.setup()  # 🔹 Ensure setup is called again

        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            print("Behavior Tree Stopped.")

    def print_tree(self):
        print(
            py_trees.display.unicode_tree(root=self._task_tree.root, show_status=True)
        )

    def plot_tree(self, output_path):
        py_trees.display.render_dot_tree(
            self._task_tree.root, target_directory=output_path
        )

    def setup_children(self):
        self._task_tree.setup()

    def get_task_completion_status(self):
        return self._blackboard.task_complete

if __name__ == "__main__":

    start_time = time.time()
    print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")


    RESNER18_MODEL_PATH = "ztests\man_test_scripts\\resnet18_rack_detection_v3.pth"                 #Capping
    # RESNER18_MODEL_PATH =  "ztests\man_test_scripts\\resnet18_chemspeed_plate_detection_40_v2.pth"    #CS

    # CLASS_MAP = {"0": "hazard","1": "no hazard"}  # Class mapping

    CLASS_MAP =['hazard', 'no hazard']

    _nav_task = SafeManipulationTask(destination_node_id = 30, kuka_sequence_id=99,
        user_name="Satheesh", vision_model_path=RESNER18_MODEL_PATH, vision_class_map=CLASS_MAP)
    _nav_task.setup_children()
    _nav_task.run()
    _nav_task.print_tree()
    _nav_task.get_task_completion_status()

    endtime = time.time()
    print(f"Script ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(endtime))}")
    print(f"Total execution time: {endtime - start_time:.2f} seconds")
    print("Script execution completed.")