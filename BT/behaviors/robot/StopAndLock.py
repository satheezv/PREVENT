#!/usr/bin/env python

import py_trees
import rospy
# from utils.kuka_send_nav_msg import KukaTaskPublisher
from utils.kuka_send_man_msg import KukaTaskPublisher

class ExecuteStopBehavior(py_trees.behaviour.Behaviour):
    """
    A PyTrees behavior that publishes a stop command to the robot.
    """

    def __init__(self, name="ExecuteStop", node_id=0, task_type=1, task_name="Emergency Stop"):
        """
        Initializes the behavior.

        :param name: Name of the behavior.
        :param node_id: Node ID where the robot should stop.
        :param task_type: Type of the task (2 for emergency stop).
        :param task_name: Name of the stop command.
        """
        super(ExecuteStopBehavior, self).__init__(name)
        self.executed = False  # ✅ Track execution to prevent duplicate messages

        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="Emergency_stop_triggered", access=py_trees.common.Access.WRITE
        )

        self._blackboard.register_key(
            key="kuka_seq_id", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(key="is_obstacle_detected", access=py_trees.common.Access.READ)
        self.kuka_publisher = KukaTaskPublisher()  # ✅ Use provided task_name
        self.node_id = node_id
        self.task_type = task_type
        self.task_name = task_name

    def initialise(self):
        self._blackboard.Emergency_stop_triggered = False
        self.executed = False

    def update(self):

        if self.name == "Mid_Stop_Robot" and not self._blackboard.is_obstacle_detected:
            print(f"No obstacle detected and gas sensor read behavior {self.name} is bypassed.")
            return py_trees.common.Status.SUCCESS
        else:
            """
            Publishes the stop command and returns the execution status.
            """
            try:
                if not rospy.core.is_initialized():
                    rospy.logwarn(f"[{self.name}] ROS node is not initialized. Cannot send stop command.")
                    return py_trees.common.Status.FAILURE
                
                if not self.executed:
                    rospy.loginfo(f"[{self.name}] Sending Stop Command to Robot...")
                    self._seq_id = self._blackboard.kuka_seq_id

                    self.kuka_publisher.publish_task(node_id=self.node_id, task_type=self.task_type, task_name=self.task_name, seq_id=self._seq_id, program_name="stop", fine_localization=False)  # ✅ Ensure stop message is published
                    self._blackboard.kuka_seq_id = self._seq_id + 1
                    self._blackboard.Emergency_stop_triggered = True
                    rospy.loginfo(f"[{self.name}] Stop Command Sent!")
                    self.executed = True  # ✅ Prevent multiple executions
                return py_trees.common.Status.SUCCESS
            except Exception as e:
                rospy.logerr(f"[{self.name}] Stop command failed: {e}")
                return py_trees.common.Status.FAILURE
