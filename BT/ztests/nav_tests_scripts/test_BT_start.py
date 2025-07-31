#!/usr/bin/env python

import py_trees
import rospy
from behaviors.robot.NavToNode import ExecuteTaskBehavior

def create_behavior_tree():
    """
    Creates a behavior tree with ExecuteTaskBehavior wrapped in a OneShot decorator
    to ensure it runs only once.
    """
    # Create the task behavior node
    task_behavior = ExecuteTaskBehavior(
        name="NavigateToNode",
        node_id=26,
        task_type=1,  # Example task type
        task_name="Nav_"
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

    behavior_tree = create_behavior_tree()

    rospy.loginfo("Starting behavior tree execution...")
    try:
        while not rospy.is_shutdown():
            behavior_tree.tick()
            print("Tick...")
            rospy.sleep(2)  # Tick the tree every 2 seconds
    except Exception as e:
        print(f"\n🛑 Error occurred: {e}")
    finally:
        print("\n🛑 Behavior Tree Stopped.")
