# import py_trees
# import time
# import rospy
# import signal
# import operator
# from datetime import datetime
# from behaviors.sensor.GetGasSensorPrediction import ReadGasSensorBehavior
# from behaviors.sensor.CaptureImage import CaptureImageBehavior
# # from behaviors.sensor.GetVisionPrediction import CLIPImageTextMatching
# from behaviors.sensor.GetVisionPrediction_MultipleImages import CLIPImageTextMatching
# from behaviors.robot.NavToNode import ExecuteTaskBehavior
# from behaviors.robot.StopAndLock import ExecuteStopBehavior
# from behaviors.robot.GetConsentAndUnlock import ExecuteUnlockBehavior
# from behaviors.robot.LocationCheck import isKukaReachedDestination
# from behaviors.communicator.SendAlertMsg import Alert
# from behaviors.terminate_tree import TerminateTree
# from behaviors.dummy_behavior import AlwaysRunningBehavior

# def create_navigation_behavior_tree(destination_node_id, user_name="Satheesh"):
#     """
#     Creates a sequence-based behavior tree that stops the robot when a hazard is detected.
#     """

#     blackboard = py_trees.blackboard.Client()
#     blackboard.register_key(
#         key="nav_skill_complete", access=py_trees.common.Access.WRITE
#     )

#     destination_node_id = destination_node_id
#     user_name = user_name

#     object_labels = ["a vial", "a glove", "a screwdriver", "a liquid spill", "an unknown object"]
#     rospy.init_node("hazard_behavior_tree", anonymous=True, disable_signals=True)  # ✅ Allow manual signal handling

#     start_robot_behavior = ExecuteTaskBehavior(node_id=destination_node_id, task_name="Navigate2Node", task_type=1, name="robotNavBehavior")


#     # ✅ Step 1: Create separate stop behaviors to avoid parent conflict
#     stop_robot_behavior_gas = ExecuteStopBehavior(name="StopRobotGas",
#         node_id=destination_node_id,
#         task_type=1,
#         task_name="stop")

#     stop_robot_behavior_vision = ExecuteStopBehavior(name="StopRobotVision",
#         node_id=destination_node_id,
#         task_type=1,
#         task_name="stop")

    
#     destination_reached = isKukaReachedDestination(name = "kukaDestinationArrivalCheck", topic_name = "/kuka2/robot_status", target_node_id = destination_node_id)

#     # ✅ Step 6: Parallel behavior to confirm arrival to the destination

#     analyse_and_decide = py_trees.composites.Parallel(
#         name="AnalyseDecideParallel",
#         policy=py_trees.common.ParallelPolicy.SuccessOnOne()  # ✅ Stops when any sequence triggers
#     )

#     analyse_and_decide.add_children([hazard_detection, destination_reached])

#     # start_robot_behavior = ExecuteTaskBehavior(node_id=24, task_name="Navigate2Node", task_type=2, name="robotNavBehavior")

#     # ✅ Step 6: top level sequence behavior
#     safe_navigation_skill = py_trees.composites.Sequence(name="SafeNavigationSkill", memory=True)

#     alert_update_user = Alert(name = "UpdateUser", user_name=user_name)

#     unlock_prepare_robot = ExecuteUnlockBehavior(name="UnlockRobot",
#         task_type=1,
#         node_id=destination_node_id,
#         task_name="unlock")

#     safe_navigation_skill.add_children([start_robot_behavior,analyse_and_decide,alert_update_user,unlock_prepare_robot])


#     nav_skill_root = py_trees.composites.Selector("root-safe_nav", memory=True)
#     nav_skill_root.add_child(
#         py_trees.behaviours.CheckBlackboardVariableValue(
#             check=py_trees.common.ComparisonExpression(
#                 "destination_reached", True, operator.eq
#             ),
#             name="is_nav_complete",
#         )
#     )

#     nav_skill_root.add_child(safe_navigation_skill)

#     return nav_skill_root


# def signal_handler(sig, frame):
#     """Handles keyboard interrupt for clean shutdown."""
#     print("Stopping Hazard Detection Behavior Tree... Exiting.")
#     rospy.signal_shutdown("Keyboard Interrupt")
#     exit(0)