import py_trees
from kmriiwa_chemist_msgs.msg import RobotStatus, TaskStatus
from utils.kuka_get_status import kukaRobotStatus

class isKukaReachedDestination(py_trees.behaviour.Behaviour):
    def __init__(self, name, topic_name, target_node_id):
        """
        Behavior Tree Node to read a ROS topic and check if the robot has reached the target location.
        :param name: Name of the behavior node
        :param topic_name: Name of the ROS topic
        :param target_node_id: Target node ID to check if the robot has reached the destination
        """
        super(isKukaReachedDestination, self).__init__(name)
        self.ros_topic_reader = kukaRobotStatus(topic_name, msg_type = RobotStatus)
        self.received_message:RobotStatus = None
        self.target_node_id = target_node_id

        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="destination_reached", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="last_graph_node_id", access=py_trees.common.Access.WRITE
        )

    def initialise(self):
        self._blackboard.destination_reached = False
        self._blackboard.last_graph_node_id = None

    def update(self):
        """Reads the latest message from the ROS topic and checks if the robot has reached the target node."""
        self.received_message = self.ros_topic_reader.get_latest_message()
        if self.received_message:
            print(f"Received RobotStatus: Timestamp: {self.received_message.stamp}, "
                  f"State: {self.received_message.robot_op_state}, "
                  f"Safety: {self.received_message.safety_state}, "
                  f"Battery: {self.received_message.state_of_charge}, "
                  f"Last Node: {self.received_message.last_graph_node_id}")
            self._blackboard.last_graph_node_id = self.received_message.last_graph_node_id
            
            if self.received_message.last_graph_node_id == self.target_node_id:
                print("Robot has reached the target location!")
                self._blackboard.destination_reached = True
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.RUNNING  # Keep waiting until target is reached
        else:
            return py_trees.common.Status.RUNNING  # Wait until a message is received

    def terminate(self, new_status):
        """Clean up when exiting the behavior."""
        pass


# main manipulation task status check
class IsKukaTaskCompleted(py_trees.behaviour.Behaviour):
    def __init__(self, name, topic_name):
        """
        Behavior Tree Node to read a ROS topic and check if the robot has reached the target location.
        :param name: Name of the behavior node
        :param topic_name: Name of the ROS topic
        :param target_node_id: Target node ID to check if the robot has reached the destination
        """
        super(IsKukaTaskCompleted, self).__init__(name)
        self.ros_topic_reader = kukaRobotStatus(topic_name, msg_type = TaskStatus)
        self.received_message:TaskStatus = None

        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="task_completed", access=py_trees.common.Access.WRITE
        )

        self._blackboard.register_key(
            key="task_status", access=py_trees.common.Access.WRITE
        )

        self._blackboard.register_key(
            key="kuka_manipulation_task_seq_id", access=py_trees.common.Access.READ
        )

        self._blackboard.register_key(
            key="main_task_initiated", access=py_trees.common.Access.READ
        )

    
        self._blackboard.task_status = []
        self._first_print = None

    def initialise(self):
        self._blackboard.task_completed = False
        self._first_print = True

    def update(self):
        if self._blackboard.main_task_initiated:
            _kukaSeqId = self._blackboard.kuka_manipulation_task_seq_id
            """Reads the latest message from the ROS topic and checks if the robot has reached the target node."""
            self.received_message = self.ros_topic_reader.get_latest_message()
            if self.received_message:
                print(f"Received RobotTaskStatus: cmd_seq: {self.received_message.cmd_seq}, "
                    f"name: {self.received_message.task_name}, "
                    f"state: {self.received_message.task_lbr_state}, ")
                self._blackboard.task_status = [self.received_message.cmd_seq, self.received_message.task_name, self.received_message.task_lbr_state]
                if self.received_message.cmd_seq == _kukaSeqId and self.received_message.task_lbr_state == TaskStatus.FINISHED:
                    print("Robot has completed the task!")
                    self._blackboard.task_completed = True
                    return py_trees.common.Status.SUCCESS
                else:
                    if self._first_print:
                        print("Task is still in progress...")
                        self._first_print = False
                    return py_trees.common.Status.RUNNING  # Keep waiting until target is reached
            else:
                print("Waiting for task status message...")
                return py_trees.common.Status.RUNNING  # Wait until a message is received
        else:
            if self._first_print:
                print("Waiting Main task not initiated ...")
            return py_trees.common.Status.RUNNING  # Wait until a message is received

    def terminate(self, new_status):
        """Clean up when exiting the behavior."""
        pass


# small manipulation executions status check

class IsKukaManipulationCompleted(py_trees.behaviour.Behaviour):
    def __init__(self, name, topic_name):
        """
        Behavior Tree Node to read a ROS topic and check if the robot has reached the target location.
        :param name: Name of the behavior node
        :param topic_name: Name of the ROS topic
        :param target_node_id: Target node ID to check if the robot has reached the destination
        """
        super(IsKukaManipulationCompleted, self).__init__(name)
        self.ros_topic_reader = kukaRobotStatus(topic_name, msg_type = TaskStatus)
        self.received_message:TaskStatus = None

        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="supportive_manipulation_completed", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="kuka_Supportive_manipulation_seq_id", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="kuka_seq_id", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(key="is_obstacle_detected", access=py_trees.common.Access.READ)
        self._first_print = None

    def initialise(self):
        self._blackboard.supportive_manipulation_completed = False
        self._first_print = True

    def update(self):
        if self.name in ["GsToCheck Status Check", "GsToBase Status Check"] and not self._blackboard.is_obstacle_detected:
            print(f"No obstacle detected and status check behavior {self.name} is bypassed.")
            return py_trees.common.Status.SUCCESS
        else:
            _kukaSeqId = self._blackboard.kuka_Supportive_manipulation_seq_id
            """Reads the latest message from the ROS topic and checks if the robot has reached the target node."""
            self.received_message = self.ros_topic_reader.get_latest_message()
            if self.received_message:
                print(f"Received Supportive RobotTaskStatus: cmd_seq: {self.received_message.cmd_seq}, "
                    f"Task_name: {self.received_message.task_name}, "
                    f"KMP_Task_State: {self.received_message.task_kmp_state},"
                    f"LBR_Task_state:{self.received_message.task_lbr_state}")
                if self.received_message.cmd_seq == _kukaSeqId and self.received_message.task_lbr_state == TaskStatus.FINISHED:
                    print("Robot has completed the task!")
                    self._blackboard.supportive_manipulation_completed = True
                    return py_trees.common.Status.SUCCESS
                else:
                    if self._first_print:
                        print("Task is still in progress...")
                        self._first_print = False
                    return py_trees.common.Status.RUNNING  # Keep waiting until target is reached           
            else:
                print("Waiting for task status message...")
                return py_trees.common.Status.RUNNING  # Wait until a message is received

    def terminate(self, new_status):
        """Clean up when exiting the behavior."""
        pass