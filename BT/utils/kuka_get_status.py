import rospy
from std_msgs.msg import String  # Change this to match the ROS topic message type
from kmriiwa_chemist_msgs.msg import RobotStatus, TaskStatus

class kukaRobotStatus:
    def __init__(self, topic_name, msg_type):
        """
        Utility class to subscribe to a ROS topic and store the latest message.
        :param topic_name: Name of the ROS topic
        :param target_node_id: Target node ID to check if the robot has reached the destination
        """
        self.topic_name = topic_name
        self.msg_type = msg_type
        # self.target_node_id = target_node_id
        self.latest_message = None

        # Initialize ROS node if it hasn't been initialized
        if not rospy.core.is_initialized():
            rospy.init_node("ros_topic_reader", anonymous=True)

        self.subscriber = rospy.Subscriber(self.topic_name, msg_type, self.callback)

    def callback(self, msg):
        """Callback function to store the latest received RobotStatus message."""
        self.latest_message = msg

    def get_latest_message(self):
        """Returns the latest received RobotStatus message."""
        return self.latest_message