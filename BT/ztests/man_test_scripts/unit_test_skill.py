import unittest
import py_trees
from skills.SafeManipulation import create_manipulation_behavior_tree
from tasks.safe_manipulation_Main import SafeManipulationTask
from py_trees.blackboard import Blackboard

class TestSafeNavigationTask(unittest.TestCase):

    def setUp(self):
        self.vision_model_path = "test_model_path"
        self.vision_class_map = {"0": "empty_floor", "1": "object_on_floor"}
        self.task = SafeManipulationTask(
            destination_node_id=29,
            user_name="TestUser",
            kuka_sequence_id=1,
            vision_model_path=self.vision_model_path,
            vision_class_map=self.vision_class_map
        )
        self.task.setup_children()

    def test_blackboard_initialization(self):
        """Test if Blackboard variables are initialized correctly."""
        self.assertFalse(self.task._blackboard.task_complete)
        self.assertEqual(self.task._blackboard.kuka_seq_id, 1)
        self.assertFalse(self.task._blackboard.restart_tree)

    def test_task_completion(self):
        """Test if task completion flag is set correctly."""
        self.task._blackboard.task_complete = True
        self.assertTrue(self.task.get_task_completion_status())

    def test_restart_tree(self):
        """Test if the behavior tree restarts correctly."""
        self.task._blackboard.restart_tree = True
        self.task.run()  # Simulate running the task
        self.assertFalse(self.task._blackboard.restart_tree)
        self.assertFalse(self.task._blackboard.task_complete)  # Tree should reset and run again

    def test_tree_structure(self):
        """Test if the behavior tree structure is correctly formed."""
        tree = create_manipulation_behavior_tree(
            self.task.destination_node_id,
            self.vision_model_path,
            self.vision_class_map,
            self.task.user_name
        )
        self.assertIsInstance(tree, py_trees.composites.Selector)

    def test_tick_behavior_tree(self):
        """Test behavior tree ticks successfully."""
        self.task._task_tree.tick()
        self.assertIn(self.task._task_tree.root.status, [
            py_trees.common.Status.RUNNING, 
            py_trees.common.Status.SUCCESS
        ])

if __name__ == "__main__":
    unittest.main()
