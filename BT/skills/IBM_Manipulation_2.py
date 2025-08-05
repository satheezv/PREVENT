import py_trees
import time
import rospy
import signal
import operator
from datetime import datetime
from behaviors.sensor.GetGasSensorPrediction import ReadGasSensorBehavior, ReadGasSensorBehavior_Once
from behaviors.sensor.VlmAnalysis_MultipleImages_man import CLIPImageTextMatching
from behaviors.sensor.GetVisionPrediction_ResNet18 import ResNetPredictionBehavior, ResNetPredictionBehavior_Once
from behaviors.robot.ArmExecution import ExecuteArmTaskBehavior
from behaviors.robot.StopAndLock import ExecuteStopBehavior
from behaviors.robot.StopUnlockRetract import ExecuteCancelUnlockRetractBehavior
from behaviors.robot.GetConsentAndUnlock import ExecuteUnlockBehavior
from behaviors.robot.StatusCheck import IsKukaTaskCompleted, IsKukaManipulationCompleted
from behaviors.robot.HandleObject import ExecuteObjectHandlingBehavior
from behaviors.communicator.SendAlertMsg import Man_Alert
from behaviors.terminate_tree import TerminateTree
from behaviors.dummy_behavior import AlwaysRunningBehavior
from utils.realSenseCamera import RealSenseCamera
from utils.GasSensorHidInterface import HIDGasSensor
from utils.slack_msging import SlackAlertUtility

def create_manipulation_behavior_tree(task_node_id, Resnet_vision_model_path, vision_class_map = None, user_name="Satheesh"):
    """
    Creates a sequence-based behavior tree that stops the robot when a hazard is detected.
    """

    # Initialize blackboard for shared data
    blackboard = py_trees.blackboard.Client()
    blackboard.register_key(
        key="manipulation_task_completed", access=py_trees.common.Access.WRITE
    ) 

    blackboard.register_key(
        key="skill_completed", access=py_trees.common.Access.READ
    )


    # Initialize task parameters
    task_node_id = task_node_id
    user_name = user_name

    # Initialize utilities and sensors
    camera = RealSenseCamera()
    SlackInterface = SlackAlertUtility(user_name=user_name)

    # Define object labels for vision monitoring
    object_labels = ["empty tray", "broken glass pieces on a tray", "Vial cap on a tray", "liquid spill on a tray", "unknown object on a tray"]  # Updated with actual labels

    # Initialize ROS node
    rospy.init_node("hazard_behavior_tree", anonymous=True, disable_signals=True)  # ✅ Allow manual signal handling

    # Define behaviors for the manipulation skill
    task_name = "rackTransfer"
    gas_sensor_reading_behavior = ReadGasSensorBehavior_Once(name="Init_reading",num_reads=3)
    stop_robot_behavior = ExecuteStopBehavior(name="Init_Stop_Robot",
        node_id=task_node_id,
        task_type=1,
        task_name="stop")

    robot_calibration_behavior = ExecuteArmTaskBehavior(node_id=task_node_id, task_name="Main Arm Task", task_type=2, name="robotManipulationBehavior", program_name="CS_CalibrationAndMoveCheck")    
    calibration_status_behavior = IsKukaManipulationCompleted(name="Calibration Status Check", topic_name="/kuka2/task_status")


    task_status_behavior = IsKukaTaskCompleted(name="Arm Task Status Check", topic_name="/kuka2/task_status")

    vision_ResNet_behavior = ResNetPredictionBehavior_Once(name="ResNetBehaviorSafeGrasp", model_path=Resnet_vision_model_path, class_map=vision_class_map, camera=camera)
    VLM_monitoring = CLIPImageTextMatching(name="ClipAnalysis", text_labels=object_labels, camera=camera)
    task_or_retract_behavior = ExecuteCancelUnlockRetractBehavior(node_id=task_node_id, task_name="PickupRack/ArmRetract", task_type=0, name="retractBehavior", program_name="CS_rackTransfer", stopUnlock_robot=True)  # Stop Unlock
    task_or_retract_status_behavior = IsKukaManipulationCompleted(name="retract Status Check", topic_name="/kuka2/task_status")

    move_gasSensor_ToCheck_behavior = ExecuteObjectHandlingBehavior(node_id=task_node_id, task_name="Pickup IoT sensor Module", task_type=0, name="Move Sensor to check Behavior", program_name="CS_sensorToCheck")
    moveGsToCheck_status_behavior = IsKukaManipulationCompleted(name="GsToCheck Status Check", topic_name="/kuka2/task_status")

    gas_sensor_reading_behavior_2 = ReadGasSensorBehavior_Once(name="Mid_reading",num_reads=3)
    move_gasSensor_ToBase_behavior = ExecuteObjectHandlingBehavior(node_id=task_node_id, task_name="Return IoT sensor Module", task_type=0, name="Move Sensor to base Behavior", program_name="CS_sensorToBase")
    moveGsToBase_status_behavior = IsKukaManipulationCompleted(name="GsToBase Status Check", topic_name="/kuka2/task_status")

    # retract_with_gs_behavior = ExecuteCancelUnlockRetractBehavior(node_id=task_node_id, task_name="Arm & Sensor Retract", task_type=2, name="retractBehaviorWithSensor", program_name="sensorToBase", stopUnlock_robot=False)  # Just retract
    # place_gasSensor_behavior = ExecuteHandlingBehavior(node_id=task_node_id, task_name="Place IoT sensor Module", task_type=2, name="retractBehavior", program_name="PlaceGasSensor")
    stop_robot_behavior_2 = ExecuteStopBehavior(name="Mid_Stop_Robot",
        node_id=task_node_id,
        task_type=0,
        task_name="stop")
    alert_update_user = Man_Alert(name="Update User", slack_alert_utility=SlackInterface)
    unlock_prepare_robot = ExecuteCancelUnlockRetractBehavior(name="Unlock Robot", node_id=task_node_id, task_name="unlock", task_type=0, program_name="unlock", slack_alert_utility=SlackInterface)  # Just unlock

    # Define sequences and selectors for the behavior tree
    gas_sensor_perception_init_seq = py_trees.composites.Sequence(name="Gas Sensor Perception initial Sequence", memory=True)
    gas_sensor_perception_init_seq.add_children([gas_sensor_reading_behavior, stop_robot_behavior])

    calibration_seq = py_trees.composites.Sequence(name="Calibration and Check Pose Sequence", memory=True)
    calibration_seq.add_children([robot_calibration_behavior,calibration_status_behavior])

    check_and_start_sel = py_trees.composites.Selector(name="Check and Start Sequence", memory=True)
    check_and_start_sel.add_children([gas_sensor_perception_init_seq, calibration_seq])

    vision_perseption_seq = py_trees.composites.Sequence(name="VLM based vision perception sequence", memory=True)
    vision_perseption_seq.add_children([vision_ResNet_behavior, VLM_monitoring, task_or_retract_behavior, task_or_retract_status_behavior])

    gas_sensor_perception_mid_seq = py_trees.composites.Sequence(name="Gas Sensor Perception mid Sequence", memory=True)
    gas_sensor_perception_mid_seq.add_children([move_gasSensor_ToCheck_behavior, moveGsToCheck_status_behavior, gas_sensor_reading_behavior_2, move_gasSensor_ToBase_behavior, moveGsToBase_status_behavior, stop_robot_behavior_2])

    hazard_perception_action_sequence = py_trees.composites.Sequence(name="Multi_Modal Hazard_perception sequence", memory=True)
    hazard_perception_action_sequence.add_children([vision_perseption_seq, gas_sensor_perception_mid_seq])

    analyse_and_decide_par = py_trees.composites.Parallel(name="check for completion/hazard", policy=py_trees.common.ParallelPolicy.SuccessOnOne())
    analyse_and_decide_par.add_children([hazard_perception_action_sequence, task_status_behavior])

    safe_grasp_skill = py_trees.composites.Sequence(name="SafeGraspSkill", memory=True)
    safe_grasp_skill.add_children([check_and_start_sel, analyse_and_decide_par, alert_update_user, unlock_prepare_robot])

    safe_man_skill_root = py_trees.composites.Selector("root-safe_man", memory=True)
    safe_man_skill_root.add_child(
        py_trees.behaviours.CheckBlackboardVariableValue(
            check=py_trees.common.ComparisonExpression(
                "skill_completed", True, operator.eq
            ),
            name="skill_complete_check",
        )
    )

    safe_man_skill_root.add_child(safe_grasp_skill)

    return safe_man_skill_root

def signal_handler(sig, frame):
    """Handles keyboard interrupt for clean shutdown."""
    print("Stopping Hazard Detection Behavior Tree... Exiting.")
    rospy.signal_shutdown("Keyboard Interrupt")
    exit(0)
