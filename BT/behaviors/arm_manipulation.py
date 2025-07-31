import py_trees
from .terminate_tree import TerminateTree
from time import sleep

class Navigate(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Navigate, self).__init__(name)
    def update(self):
        print("Navigating...")
        sleep(3)
        return py_trees.common.Status.SUCCESS