#!/usr/bin/env python

import py_trees
import rospy
import time
from utils.kuka_send_nav_msg import KukaTaskPublisher
from utils.GasSensorHidInterface import HIDGasSensor
from utils.slack_msging import SlackAlertUtility

class ExecuteUnlockBehavior(py_trees.behaviour.Behaviour):
    """
    A PyTrees behavior that publishes a robot task command.
    """

    def __init__(self, name, task_type, task_name, node_id=None, sensor = None, slack_alert_utility:SlackAlertUtility=None):
        """
        Initializes the behavior.

        :param name: Name of the behavior.
        :param node_id: Node ID where the robot should navigate.
        :param task_type: Type of the task (e.g., movement, manipulation).
        :param task_name: Name of the task.
        """
        super(ExecuteUnlockBehavior, self).__init__(name)

        self.gas_sensor:HIDGasSensor = sensor
        self.duration = 1
        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="kuka_seq_id", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="last_graph_node_id", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="destination_reached", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="Emergency_stop_triggered", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="restart_tree", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="is_VOC_hazard_detected", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="identified_hazard_infront", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(key="is_obstacle_detected", access=py_trees.common.Access.READ)
        

        # if self._blackboard.exists("last_graph_node_id"):
        #     node_id = self._blackboard.last_graph_node_id
        # else:
        node_id = node_id
        task_name = task_name

        self._seq_id = self._blackboard.kuka_seq_id
        self.kuka_unlock_publisher = KukaTaskPublisher(node_id, task_type, task_name=task_name, seq_id=self._seq_id)
        self.slack_alert_utility = slack_alert_utility
        # self.kuka_resume_publisher = KukaTaskPublisher(node_id, task_type=1, task_name="Resume", seq_id=self._seq_id+1)
    
    def initialise(self):
        self._First_print = True

    def update(self):
        """
        Publishes the task command and returns the execution status.
        """
        try:
            if self._blackboard.exists("destination_reached"):   
                if self._blackboard.destination_reached:
                    rospy.loginfo(f"[{self.name}] Robot has reached the destination.")
                    return py_trees.common.Status.SUCCESS
            if self._blackboard.exists("Emergency_stop_triggered"):
                if self._blackboard.Emergency_stop_triggered:
                    if self._blackboard.is_VOC_hazard_detected and self._blackboard.identified_hazard_infront == "an unknown object":
                        print("🚨 VOC Hazard detected! Robot is locked.")
                        start_time = time.time()  # Record the start time
                        duration = 5 * 60  # 5 minutes in seconds
                        while time.time() - start_time < duration:
                            gas_sensor_reading = self.gas_sensor.read_multiple(num_reads=self.num_reads)
                            if gas_sensor_reading["voc"] < self.VOC_THRESHOLD:
                                self.kuka_unlock_publisher.publish_task()
                                rospy.sleep(10)
                                # self.kuka_resume_publisher.publish_task()
                                self._blackboard.Emergency_stop_triggered = False
                                self._blackboard.kuka_seq_id = self._seq_id + 1
                                print(f"New Seq ID: {self._blackboard.kuka_seq_id}")
                                self._blackboard.restart_tree = True
                                return py_trees.common.Status.SUCCESS
                            time.sleep(1)  # Optional: Prevents excessive CPU usage, adjust as needed
                    else:
                        # userconsent = input("Do you want to unlock the robot? (yes/no): ").strip().lower()
                        userconsent = self.slack_alert_utility.read_latest_message()
                        if self._First_print:
                            print("waiting for user consent, please type 'yes' or 'resume' from the authorized slack User-id to unlock and resume the operation")
                            self._First_print = False
                        if userconsent in ["yes", "y", "continue", "resume"]:
                            print("received the message: ", userconsent)
                            print("🔓 Unlocking the robot...")
                            self.kuka_unlock_publisher.publish_task()
                            rospy.sleep(5)
                            # self.kuka_resume_publisher.publish_task()
                            self._blackboard.Emergency_stop_triggered = False
                            self._blackboard.kuka_seq_id = self._seq_id + 1
                            print(f"New Seq ID: {self._blackboard.kuka_seq_id}")
                            self._blackboard.restart_tree = True
                            return py_trees.common.Status.SUCCESS
                        else:
                            # print("❌ Unlock operation canceled.")
                            self.kuka_publisher = None
                            return py_trees.common.Status.RUNNING
        except Exception as e:
            rospy.logerr(f"[{self.name}] Task execution failed: {e}")
            return py_trees.common.Status.FAILURE

    