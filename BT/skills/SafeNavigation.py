import py_trees
import time
import rospy
import signal
import operator
from datetime import datetime
from behaviors.sensor.GetGasSensorPrediction import ReadGasSensorBehavior
from behaviors.sensor.CaptureImage import CaptureImageBehavior
# from behaviors.sensor.GetVisionPrediction import CLIPImageTextMatching
from behaviors.sensor.VlmAnalysis_MultipleImages import CLIPImageTextMatching
from behaviors.sensor.GetVisionPrediction_ResNet18 import ResNetPredictionBehavior
from behaviors.robot.NavToNode import ExecuteTaskBehavior
from behaviors.robot.StopAndLock import ExecuteStopBehavior
from behaviors.robot.GetConsentAndUnlock import ExecuteUnlockBehavior
from behaviors.robot.StatusCheck import isKukaReachedDestination
from behaviors.communicator.SendAlertMsg import Nav_Alert
from behaviors.terminate_tree import TerminateTree
from behaviors.dummy_behavior import AlwaysRunningBehavior
from utils.realSenseCamera import RealSenseCamera
from utils.GasSensorHidInterface import HIDGasSensor
from utils.slack_msging import SlackAlertUtility


def create_navigation_behavior_tree(destination_node_id, vision_model_path, vision_class_map, user_name="Satheesh"):
    """
    Creates a sequence-based behavior tree that stops the robot when a hazard is detected.
    """

    blackboard = py_trees.blackboard.Client()
    blackboard.register_key(
        key="destination_reached", access=py_trees.common.Access.WRITE
    )

    destination_node_id = destination_node_id
    user_name = user_name

    camera = RealSenseCamera()
    GasSensor = HIDGasSensor()
    SlackInterface = SlackAlertUtility(user_name=user_name)

    # object_labels = ["a vial", "a glove", "a screwdriver", "a liquid spill", "a tissue paper", "an unknown object"]
    object_labels = [
    "chemspeed clear", "chemspeed hazard",
    "capping clear", "capping spillage", "capping obstruction", "capping missing cap", "capping missing vial",
    "navigation object glove", "navigation empty floor", "navigation other objects", "navigation object tool", "navigation object vial"
]
    rospy.init_node("hazard_behavior_tree", anonymous=True, disable_signals=True)  # ✅ Allow manual signal handling

    start_robot_behavior = ExecuteTaskBehavior(node_id=destination_node_id, task_name="Navigate2Node", task_type=1, name="robotNavBehavior")


    # ✅ Step 1: Create separate stop behaviors to avoid parent conflict
    stop_robot_behavior_gas = ExecuteStopBehavior(name="StopRobotGas",
        node_id=destination_node_id,
        task_type=1,
        task_name="stop")

    stop_robot_behavior_vision = ExecuteStopBehavior(name="StopRobotVision",
        node_id=destination_node_id,
        task_type=1,
        task_name="stop")

    # ✅ Step 2: Hazard detection sequences
    gas_sensor_sequence = py_trees.composites.Sequence(name="GasSensorSequence", memory=True)
    vision_sensor_sequence = py_trees.composites.Sequence(name="VisionSensorSequence", memory=True)

    # ✅ Step 3: Create sensor behaviors
    gas_sensor_behavior = ReadGasSensorBehavior(num_reads=3, sensor=GasSensor)

    # capture_behavior = CaptureImageBehavior(name="Capture Image")
    vision_sensor_behavior = ResNetPredictionBehavior(name="Resnet Binary Prediction", model_path=vision_model_path, class_map=vision_class_map, camera=camera)
    DummyAlwaysRunningBehavior = AlwaysRunningBehavior(name="AlwaysRunningBehavior")
    # ✅ Step 4: If a hazard is detected, stop the robot
    gas_sensor_sequence.add_children([gas_sensor_behavior, stop_robot_behavior_gas])
    # gas_sensor_sequence.add_children([DummyAlwaysRunningBehavior, stop_robot_behavior_gas])

    vision_sensor_sequence.add_children([vision_sensor_behavior, stop_robot_behavior_vision])
    # vision_sensor_sequence.add_children([DummyAlwaysRunningBehavior, stop_robot_behavior_vision])

    # ✅ Step 5: Parallel behavior to monitor both sensors
    hazard_detection_and_stop = py_trees.composites.Parallel(
        name="HazardDetectionParallel",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()  # ✅ Stops when any sequence triggers
    )
    hazard_detection_and_stop.add_children([gas_sensor_sequence, vision_sensor_sequence])

    analyse_and_generate_msg = CLIPImageTextMatching(name="AnalyseAndGenerateMsg", text_labels=object_labels, camera=camera)

    perception_and_report = py_trees.composites.Sequence(name="PerceptionAndReport", memory=True)

    perception_and_report.add_children([hazard_detection_and_stop, analyse_and_generate_msg])

    destination_reached = isKukaReachedDestination(name = "kukaDestinationArrivalCheck", topic_name = "/kuka2/robot_status", target_node_id = destination_node_id)

    # ✅ Step 6: Parallel behavior to confirm arrival to the destination

    analyse_and_decide = py_trees.composites.Parallel(
        name="AnalyseDecideParallel",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()  # ✅ Stops when any sequence triggers
    )

    analyse_and_decide.add_children([perception_and_report, destination_reached])

    # ✅ Step 6: top level sequence behavior
    safe_navigation_skill = py_trees.composites.Sequence(name="SafeNavigationSkill", memory=True)

    alert_update_user = Nav_Alert(name = "UpdateUser", slack_alert_utility=SlackInterface)

    unlock_prepare_robot = ExecuteUnlockBehavior(name="UnlockRobot",
        task_type=1,
        node_id=destination_node_id,
        task_name="unlock",
        sensor=GasSensor,
        slack_alert_utility=SlackInterface)

    safe_navigation_skill.add_children([start_robot_behavior,analyse_and_decide,alert_update_user,unlock_prepare_robot])


    nav_skill_root = py_trees.composites.Selector("root-safe_nav", memory=True)
    nav_skill_root.add_child(
        py_trees.behaviours.CheckBlackboardVariableValue(
            check=py_trees.common.ComparisonExpression(
                "destination_reached", True, operator.eq
            ),
            name="is_nav_complete",
        )
    )

    nav_skill_root.add_child(safe_navigation_skill)

    return nav_skill_root


def signal_handler(sig, frame):
    """Handles keyboard interrupt for clean shutdown."""
    print("Stopping Hazard Detection Behavior Tree... Exiting.")
    rospy.signal_shutdown("Keyboard Interrupt")
    exit(0)