import rospy
from kmriiwa_chemist_msgs.msg import RobotCommand, LBRCommand, KMPCommand
from datetime import datetime

class KukaTaskPublisher:
    """
    Utility class for publishing robot task commands to the /kuka2/commands topic.
    """

    def __init__(self, node_id, task_type, task_name, seq_id):
        """
        Initializes the publisher.
        """
        self.node_id = node_id
        self.task_type = task_type
        self.task_name = task_name
        self.seq_id = seq_id
        self.publisher = rospy.Publisher("/kuka2/commands", RobotCommand, queue_size=10)

        # ✅ Ensure ROS node is initialized
        if not rospy.core.is_initialized():
            rospy.init_node("kuka_nav_task_pub_node", anonymous=True, disable_signals=True)  # ✅ Allow manual signal handling
            rospy.sleep(2)  # Allow time for publisher registration

    def get_seq_id(self):
        # Get current hour and minute
        now = datetime.now()
        hour = now.hour  # e.g., 14 for 2 PM
        minute = now.minute  # e.g., 45
        seconds = now.second
        # Combine hour and minute into a single integer (HHMM format)
        return hour * 100 + minute * 100 + seconds

    def publish_task(self):
        """
        Publishes a stop command to the robot.
        """
        if not rospy.core.is_initialized():
            rospy.logwarn("Cannot publish message: ROS node is not initialized.")
            return

        kuka_cmd = RobotCommand()
        kuka_cmd.seq = self.seq_id  # Unique ID
        kuka_cmd.priority_task = True
        kuka_cmd.task_type = self.task_type
        kuka_cmd.task_name = self.task_name

        arm_command = LBRCommand()
        base_command = KMPCommand()

        # ✅ Stop robot movement by setting an emergency stop task
        base_command.graph_id = 1
        base_command.node_id = self.node_id
        base_command.fine_localization = True

        arm_command.program_name = "__test__"  # Custom stop command
        arm_command.program_parameters = []

        kuka_cmd.arm_command = arm_command
        kuka_cmd.base_command = base_command

        # ✅ Publish stop command
        for i in range(3):
            self.publisher.publish(kuka_cmd)
            rospy.loginfo(f"Published Nav command: {kuka_cmd.task_name}, to node: {kuka_cmd.base_command.node_id}")
            rospy.loginfo(f"generated seq id: {kuka_cmd.seq}")
            rospy.sleep(1)