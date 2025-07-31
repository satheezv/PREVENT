import py_trees
from behaviors.navigation import Navigate
from behaviors.detection_image import DetectHazard
from behaviors.confirmation import ConfirmHazard
from behaviors.alert import Alert
from time import sleep

class TwoStageDetection:
    def __init__(self, labels, label_to_detect, slack_token, slack_user_id):
        self.labels = labels
        self.label_to_detect = label_to_detect
        self.slack_token = slack_token
        self.slack_user_id = slack_user_id
        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="terminate_tree", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="task_complete", access=py_trees.common.Access.WRITE
        )
        self._blackboard.task_complete = False
        self._blackboard.terminate_tree = False

        self.tree = self.create_hazard_detection_tree()

    def create_hazard_detection_tree(self):
        root = py_trees.composites.Selector("Hazard Detection System", memory=False)
        navigate = Navigate("Navigate")
        detect_hazard_sequence = py_trees.composites.Sequence("Detect Hazard Sequence", memory=True)
        detect_hazard = DetectHazard("Detect Hazard", self.labels, self.label_to_detect)
        confirm_hazard = ConfirmHazard("Confirm Hazard")
        # Alert behavior now handles Slack messaging
        alert = Alert("Alert", slack_token=self.slack_token, slack_user_id=self.slack_user_id, msg_to_send=self.label_to_detect)

        set_task_complete = py_trees.behaviours.SetBlackboardVariable(
            name="Set Task Complete",
            variable_name="task_complete",
            variable_value=True,
            overwrite=True,
        )
        detect_hazard_sequence.add_children([detect_hazard, confirm_hazard, alert, set_task_complete])
        root.add_children([detect_hazard_sequence, navigate])

        return root

    def run(self):
        behaviour_tree = py_trees.trees.BehaviourTree(self.tree)
        behaviour_tree.setup(timeout=15)

        # Generate the DOT file using the root node managed by BehaviourTree
        dot_graph = py_trees.display.ascii_tree(self.tree)
        with open("hazard_detection_system.dot", "w", encoding="utf-8") as file:
            file.write(dot_graph)

        while (
            not self._blackboard.task_complete and not self._blackboard.terminate_tree
        ):
            try:
                behaviour_tree.tick()
                print("tick")
                sleep(1)
            except KeyboardInterrupt:
                break

        print(
            py_trees.display.unicode_tree(root=behaviour_tree.root, show_status=True)
        )



if __name__ == "__main__":
    # labels = ["vial on the floor", "liquid spillage", "solid powder spillage", 
    #           "sharp needle", "broken glass", "empty floor", "scissor"]

    # labels = ["open vial", "closed vial", "semi-sealed vial", "liquid inside vial", "powder inside vial"]
    # labels = ["glass vial", "liquid", "powder", "knife"]
    labels = ["rack with 4 vials", "rack missing vials"]
    label_to_detect = "vial open"  # Example label to detect

    # Replace with your Slack token and channel
    SLACK_TOKEN = "xoxb-638030615362-8306647721012-l6SPuh6GTe8bTaIjMCNEEbS3"
    SLACK_USER_ID = "U04N321A5Q9"  #satheesh
    # SLACK_USER_ID = "U05Q6CC2MMM"    # francisco
    detection_system = TwoStageDetection(labels, label_to_detect, SLACK_TOKEN, SLACK_USER_ID)
    detection_system.run()
