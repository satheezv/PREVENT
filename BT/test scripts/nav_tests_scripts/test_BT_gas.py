import py_trees
import time
import rospy
import signal
from datetime import datetime
from behaviors.sensor.GetGasSensorPrediction import ReadGasSensorBehavior
from behaviors.sensor.GetVisionPrediction import VisionSensorBehavior
from behaviors.robot.NavToNode import ExecuteTaskBehavior
from behaviors.robot.StopAndLock import ExecuteStopBehavior
from behaviors.robot.GetConsentAndUnlock import ExecuteUnlockBehavior
from behaviors.robot.StatusCheck import isKukaReachedDestination
from behaviors.communicator.SendAlertMsg import Alert

def create_behavior_tree():
    """
    Creates a sequence-based behavior tree that stops the robot when a hazard is detected.
    """
    destination_node_id = 29
    safe_node_id = 28
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
    gas_sensor_behavior = ReadGasSensorBehavior(num_reads=5)
    vision_sensor_behavior = VisionSensorBehavior()

    # ✅ Step 4: If a hazard is detected, stop the robot
    gas_sensor_sequence.add_children([gas_sensor_behavior, stop_robot_behavior_gas])
    vision_sensor_sequence.add_children([vision_sensor_behavior, stop_robot_behavior_vision])

    # ✅ Step 5: Parallel behavior to monitor both sensors
    hazard_detection = py_trees.composites.Parallel(
        name="HazardDetectionParallel",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()  # ✅ Stops when any sequence triggers
    )
    hazard_detection.add_children([gas_sensor_sequence, vision_sensor_sequence])

    destination_reached = isKukaReachedDestination(name = "kukaDestinationArrivalCheck", topic_name = "/kuka2/robot_status", target_node_id = destination_node_id)

    # ✅ Step 6: Parallel behavior to confirm arrival to the destination

    analyse_and_decide = py_trees.composites.Parallel(
        name="AnalyseDecideParallel",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()  # ✅ Stops when any sequence triggers
    )

    analyse_and_decide.add_children([hazard_detection, destination_reached])

    # start_robot_behavior = ExecuteTaskBehavior(node_id=24, task_name="Navigate2Node", task_type=2, name="robotNavBehavior")

    # ✅ Step 6: top level sequence behavior
    safe_navigation_skill = py_trees.composites.Sequence(name="SafeNavigationSkill", memory=True)

    alert_update_user = Alert(name = "UpdateUser",msg_to_send=f"Hazard detected enroute to nodeid:{destination_node_id}", user_name="Satheesh")

    unlock_prepare_robot = ExecuteUnlockBehavior(name="UnlockRobot",
        node_id=safe_node_id,
        task_type=1,
        task_name="unlock")

    safe_navigation_skill.add_children([start_robot_behavior,analyse_and_decide,alert_update_user,unlock_prepare_robot])

    return py_trees.trees.BehaviourTree(safe_navigation_skill)


def signal_handler(sig, frame):
    """Handles keyboard interrupt for clean shutdown."""
    print("Stopping Hazard Detection Behavior Tree... Exiting.")
    rospy.signal_shutdown("Keyboard Interrupt")
    exit(0)


if __name__ == '__main__':
    print("Starting Hazard Detection Behavior Tree... Press Ctrl+C to exit.")

    behavior_tree = create_behavior_tree()
    signal.signal(signal.SIGINT, signal_handler)  # ✅ Capture Ctrl+C

    try:
        while not rospy.is_shutdown():
            behavior_tree.tick()
            print("Tick...")
            time.sleep(2)  # Tick the tree every 2 seconds
        print(
        py_trees.display.unicode_tree(root=behavior_tree, show_status=True)
        )
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("Behavior Tree Stopped.")
