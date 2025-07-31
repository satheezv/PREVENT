#!/usr/bin/env python

import py_trees
import rospy
from behaviors.robot.NavToNode import ExecuteTaskBehavior
from utils.kuka_send_stop_msg import KukaStopPublisher

def create_behavior_tree():
    """
    Creates a behavior tree with ExecuteTaskBehavior wrapped in a OneShot decorator
    to ensure it runs only once.
    """
    # Create the task behavior node
    task_behavior = ExecuteTaskBehavior(
        name="NavigateToNode",
        node_id=24,
        task_type=1,  # Example task type
        task_name="unlock"
    )

    # ✅ Fix: Provide "name" and "policy" arguments for OneShot
    root = py_trees.decorators.OneShot(
        name="OneShot-NavigateToNode",  # Unique name
        child=task_behavior,
        policy=py_trees.common.OneShotPolicy.ON_COMPLETION # Ensures it runs once and stops
    )

    return py_trees.trees.BehaviourTree(root)

if __name__ == '__main__':
    rospy.init_node('task_behavior_tree', anonymous=True)

    kuka_publisher = KukaStopPublisher(28, 1, "unlock")  # ✅ Use provided task_name
    kuka_publisher.publish_task()