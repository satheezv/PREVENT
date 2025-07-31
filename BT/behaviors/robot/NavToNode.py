#!/usr/bin/env python

import py_trees
import rospy
from utils.kuka_send_nav_msg import KukaTaskPublisher

class ExecuteTaskBehavior(py_trees.behaviour.Behaviour):
    """
    A PyTrees behavior that publishes a robot task command.
    """

    def __init__(self, name, node_id, task_type, task_name):
        """
        Initializes the behavior.

        :param name: Name of the behavior.
        :param node_id: Node ID where the robot should stop.
        :param task_type: Type of the task (2 for emergency stop).
        :param task_name: Name of the stop command.
        """
        super(ExecuteTaskBehavior, self).__init__(name)
        self._blackboard = py_trees.blackboard.Client()

        self._blackboard.register_key(
            key="kuka_seq_id", access=py_trees.common.Access.WRITE
        )

        self._blackboard.register_key(
            key="kuka_current_node_id", access=py_trees.common.Access.WRITE
        )

        self._seq_id = self._blackboard.kuka_seq_id

        self.kuka_nav_publisher = KukaTaskPublisher(node_id, task_type, task_name, self._seq_id)  # ✅ Use provided task_name
        self.executed = False  # ✅ Track execution to prevent duplicate messages

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
            
            if not self.executed:
                rospy.loginfo(f"[{self.name}] Sending Nav Command to Robot...")
                self.kuka_nav_publisher.publish_task()  # ✅ Ensure stop message is published
                self._blackboard.kuka_seq_id = self._seq_id + 1
                print(f"New Seq ID: {self._blackboard.kuka_seq_id}")
                rospy.loginfo(f"[{self.name}] Nav Command Sent!")
                self.executed = True  # ✅ Prevent multiple executions
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            rospy.logerr(f"[{self.name}] Nav command failed: {e}")
            return py_trees.common.Status.FAILURE
