#!/usr/bin/env python

import py_trees
import rospy
from utils.kuka_send_man_msg import KukaTaskPublisher

class ExecuteObjectHandlingBehavior(py_trees.behaviour.Behaviour):
    """
    A PyTrees behavior that publishes a robot task command.
    """

    def __init__(self, name, node_id, task_type, task_name, program_name):
        """
        Initializes the behavior.

        :param name: Name of the behavior.
        :param node_id: Node ID where the robot should stop.
        :param task_type: Type of the task (2 for emergency stop).
        :param task_name: Name of the stop command.
        """
        super(ExecuteObjectHandlingBehavior, self).__init__(name)
        self._blackboard = py_trees.blackboard.Client()

        self._blackboard.register_key(
            key="kuka_seq_id", access=py_trees.common.Access.WRITE
        )

        self._blackboard.register_key(
            key="sensor_in_hand", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="kuka_Supportive_manipulation_seq_id", access=py_trees.common.Access.WRITE
        )

        self._blackboard.register_key(key="is_obstacle_detected", access=py_trees.common.Access.READ)
        self.node_id = node_id
        self.task_type = task_type
        self.task_name = task_name
        self.program_name = program_name

        self.kuka_publisher = KukaTaskPublisher()  # ✅ Use provided task_name
        self.executed = False  # ✅ Track execution to prevent duplicate messages
        self._blackboard.sensor_in_hand = None

    def initialise(self):
        self.executed = False
        

    def update(self):
        """
        Publishes the stop command and returns the execution status.
        """
        try:
            if not rospy.core.is_initialized():
                rospy.logwarn(f"[{self.name}] ROS node is not initialized. Cannot send stop command.")
                return py_trees.common.Status.FAILURE
            if self._blackboard.is_obstacle_detected:
                if not self.executed:
                    rospy.loginfo(f"[{self.name}] Sending Manipulation Command to Robot...")
                    self._seq_id = self._blackboard.kuka_seq_id
                    self.kuka_publisher.publish_task(node_id=self.node_id, task_type=self.task_type, task_name=self.task_name, seq_id=self._seq_id, program_name=self.program_name, fine_localization=False)  # ✅ Ensure stop message is published
                    self._blackboard.kuka_Supportive_manipulation_seq_id = self._seq_id
                    self._blackboard.kuka_seq_id = self._seq_id + 1
                    print(f"New Seq ID: {self._blackboard.kuka_seq_id}")
                    rospy.loginfo(f"[{self.name}] Manipulation Command Sent!")
                    if self.program_name == "sensorToCheck":
                        self._blackboard.sensor_in_hand = True
                    elif self.program_name == "sensorToBase":
                        self._blackboard.sensor_in_hand = False
                    self.executed = True  # ✅ Prevent multiple executions
                return py_trees.common.Status.SUCCESS
            else: 
                return py_trees.common.Status.RUNNING
        except Exception as e:
            rospy.logerr(f"[{self.name}] Manipulation command failed: {e}")
            return py_trees.common.Status.FAILURE
