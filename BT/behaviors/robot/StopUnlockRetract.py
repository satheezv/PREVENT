#!/usr/bin/env python

import py_trees
import rospy
from utils.kuka_send_man_msg import KukaTaskPublisher

class ExecuteCancelUnlockRetractBehavior(py_trees.behaviour.Behaviour):
    """
    A PyTrees behavior that publishes a stop command to the robot.
    """

    def __init__(self, name="", node_id=0, task_type=0, task_name="", program_name="", stopUnlock_robot=True, slack_alert_utility = None):
        """
        Initializes the behavior.

        :param name: Name of the behavior.
        :param node_id: Node ID where the robot should stop.
        :param task_type: Type of the task (2 for emergency stop).
        :param task_name: Name of the stop command.
        """
        super(ExecuteCancelUnlockRetractBehavior, self).__init__(name)
        self.executed = False  # ✅ Track execution to prevent duplicate messages
        self.node_id = node_id
        self.name = name
        self.task_type = task_type
        self.task_name = task_name
        self.program_name = program_name
        self.stopUnlock_robot = stopUnlock_robot
        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="retract_motion_triggered", access=py_trees.common.Access.WRITE
        )

        self._blackboard.register_key(
            key="main_task_initiated", access=py_trees.common.Access.WRITE
        )

        self._blackboard.register_key(
            key="task_completed", access=py_trees.common.Access.READ
        )

        self._blackboard.register_key(
            key="kuka_seq_id", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="kuka_manipulation_task_seq_id", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="restart_tree", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="kuka_Supportive_manipulation_seq_id", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(key="is_obstacle_detected", access=py_trees.common.Access.READ)

        
        self.kuka_man_publisher = KukaTaskPublisher()  # ✅ Use provided task_name
        self.slack_alert_utility = slack_alert_utility 

    def initialise(self):
        self._blackboard.retract_motion_triggered = False
        self._blackboard.main_task_initiated = False
        self.executed = False
        self._First_print = False
        self._seq_id = self._blackboard.kuka_seq_id

    def update(self):
        """
        Publishes the stop command and returns the execution status.
        """
        try:              
            if not rospy.core.is_initialized():
                rospy.logwarn(f"[{self.name}] ROS node is not initialized. Cannot send stop command.")
                return py_trees.common.Status.FAILURE
            
            if self._blackboard.exists("task_completed"): 
                if self._blackboard.task_completed:
                    rospy.loginfo(f"[{self.name}] Robot has Completed the task successfully.")
                    return py_trees.common.Status.SUCCESS
            
            if not self.executed:
                if self.name == "Unlock Robot":
                    userconsent = self.slack_alert_utility.read_latest_message()
                    if self._First_print:
                        print("waiting for user consent, please type 'yes' or 'resume' from the authorized slack User-id to unlock and resume the operation")
                        self._First_print = False
                    if userconsent in ["yes", "y", "continue", "resume"]:
                        print("received the message: ", userconsent)
                        print("🔓 Unlocking the robot...")
                        rospy.loginfo(f"[{self.name}] Sending unlock Command to Robot...")
                        self._seq_id = self._blackboard.kuka_seq_id
                        self.kuka_man_publisher.publish_task(self.node_id, self.task_type, self.task_name, self._seq_id, program_name="Unlock Robot", fine_localization=True)  # ✅ Ensure stop message is published
                        self._seq_id += 1  # ✅ Increment sequence ID
                        self._blackboard.kuka_seq_id = self._seq_id
                        print(f"New Seq ID: {self._blackboard.kuka_seq_id}")
                        self._blackboard.restart_tree = True
                        return py_trees.common.Status.SUCCESS
                    else:
                        # print("❌ Unlock operation canceled.")
                        self.kuka_publisher = None
                        return py_trees.common.Status.RUNNING
                else:
                    if self._blackboard.is_obstacle_detected:
                        self._seq_id = self._blackboard.kuka_seq_id
                        self.kuka_man_publisher.publish_task(self.node_id, self.task_type, self.task_name, self._seq_id, program_name="retractToDrive", fine_localization=True)  # ✅ execute Just Retract program
                        self._blackboard.kuka_Supportive_manipulation_seq_id = self._seq_id
                        print(f"Executing retract task (Supportive_manipulation) due to obstacle detection with Seq ID: {self._seq_id}")
                        self._seq_id += 1  # ✅ Increment sequence ID
                        self._blackboard.kuka_seq_id = self._seq_id
                        self._blackboard.retract_motion_triggered = True
                        rospy.loginfo(f"[{self.name}] Stop, Unlock and Retract Commands Sent!")
                        self.executed = True  # ✅ Prevent multiple executions
                        return py_trees.common.Status.SUCCESS
                    else:
                        self._seq_id = self._blackboard.kuka_seq_id
                        self.kuka_man_publisher.publish_task(self.node_id, self.task_type, self.task_name, self._seq_id, program_name=self.program_name, fine_localization=True)  # ✅ execute main task
                        self._blackboard.kuka_manipulation_task_seq_id = self._seq_id
                        self._seq_id += 1  # ✅ Increment sequence ID
                        self._blackboard.kuka_seq_id = self._seq_id 
                        #main task seq id
                        self._blackboard.main_task_initiated = True
                        rospy.loginfo(f"[{self.name}] Main task {self.program_name} Command Sent!")
                        return py_trees.common.Status.SUCCESS
        except Exception as e:
            rospy.logerr(f"[{self.name}] Stop command failed: {e}")
            return py_trees.common.Status.FAILURE
