import unittest
import py_trees
from ztests.test_BT_gas import create_behavior_tree

class TestHazardBehaviorTree(unittest.TestCase):
    def setUp(self):
        """Set up the behavior tree for testing."""
        self.behavior_tree = create_behavior_tree()

    def test_behavior_tree_initialization(self):
        """Test if the behavior tree initializes correctly."""
        self.assertIsInstance(self.behavior_tree, py_trees.trees.BehaviourTree)
        print("✅ Behavior tree initialized successfully.")
    
    def test_behavior_tree_tick(self):
        """Test if the behavior tree ticks without errors."""
        try:
            self.behavior_tree.tick()
            print("✅ Behavior tree ticked successfully.")
        except Exception as e:
            self.fail(f"Behavior tree tick failed: {e}")
    
    def test_print_behavior_tree(self):
        """Prints the behavior tree structure for debugging."""
        tree_structure = py_trees.display.unicode_tree(self.behavior_tree.root, show_status=True)
        print(tree_structure)
        print("✅ Behavior tree structure printed.")

if __name__ == '__main__':
    unittest.main()
