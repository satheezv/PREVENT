import py_trees
from behaviors.navigation import Navigate
from behaviors.detection import DetectHazard
from behaviors.confirmation import ConfirmHazard
from behaviors.alert import Alert
from time import sleep

def create_hazard_detection_tree(labels, label_to_detect):
    root = py_trees.composites.Selector("Hazard Detection System", memory=False)
    navigate = Navigate("Navigate")
    detect_hazard_sequence = py_trees.composites.Sequence("Detect Hazard Sequence", memory=True)
    detect_hazard = DetectHazard("Detect Hazard", labels, label_to_detect)
    confirm_hazard = ConfirmHazard("Confirm Hazard")

    alert = Alert("Alert")
    detect_hazard_sequence.add_children([detect_hazard, confirm_hazard, alert])
    root.add_children([detect_hazard_sequence, navigate])
    return root

if __name__ == "__main__":
    labels = ["vial on the floor", "liquid spillage", "solid powder spillage", 
                "sharp needle", "broken glass", "empty floor", "scissor"]
    label_to_detect = "scissor"  # Example label to detect

    _blackboard = py_trees.blackboard.Client()
    _blackboard.register_key(
        key="terminate_tree", access=py_trees.common.Access.WRITE
    )
    _blackboard.register_key(
        key="task_complete", access=py_trees.common.Access.WRITE
    )
    _blackboard.task_complete = False
    _blackboard.terminate_tree = False
    tree = create_hazard_detection_tree(labels, label_to_detect)

    tree.add_child(
        py_trees.behaviours.SetBlackboardVariable(
            name="task_complete",
            variable_name="task_complete",
            variable_value=True,
            overwrite=True,
        ))

    dot_graph = py_trees.display.ascii_tree(tree)
    with open("hazard_detection_system.dot", "w", encoding="utf-8") as file:
        file.write(dot_graph)
    behaviour_tree = py_trees.trees.BehaviourTree(tree)
    behaviour_tree.setup(timeout=15)
    # for i in range(5):
    #     behaviour_tree.tick()
    #     print("\n")

    while (
        not _blackboard.task_complete and not _blackboard.terminate_tree
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