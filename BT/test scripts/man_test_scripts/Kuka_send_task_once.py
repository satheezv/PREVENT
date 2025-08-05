#!/usr/bin/env python

from kmriiwa_chemist_msgs.msg import RobotCommand, LBRCommand, KMPCommand, RobotStatus
import rospy
import datetime

def publish_ros_msg():

    rospy.init_node("kuka_task_pub_node", anonymous=True)
    pub = rospy.Publisher("/kuka2/commands", RobotCommand, queue_size=10)
    # sub = rospy.Subscriber("/kuka4/robot_status", RobotStatus, status)
    rospy.sleep(5)

    rospy.loginfo("node initiated")

    _seq = 1
    node = 30
    kuka_cmd = RobotCommand()
    kuka_cmd.seq = _seq
    kuka_cmd.priority_task = False
    kuka_cmd.task_type = 2
    kuka_cmd.task_name = f"nav to node: {node}"

    arm_command = LBRCommand()
    base_command = KMPCommand()

    
    base_command.graph_id = 1
    base_command.node_id = node
    base_command.fine_localization = True

    arm_command.program_name = "CalibrationAndMoveCheck"
    arm_command.program_parameters = []

    kuka_cmd.arm_command = arm_command
    kuka_cmd.base_command = base_command
    for i in range(3):
        pub.publish(kuka_cmd)
    _time = datetime.datetime.now()
    print(f"publishing command (task {_seq})", str(_time))
    


if __name__ == '__main__':
    try:
        publish_ros_msg()
    except KeyboardInterrupt:
        pass

