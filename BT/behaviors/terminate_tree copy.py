import py_trees

class TerminateTree(py_trees.behaviour.Behaviour):
    def __init__(self):
        super(TerminateTree, self).__init__('Terminate_tree')
        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key('terminate_tree', py_trees.common.Access.WRITE)
        self._blackboard.terminate_tree = False

    def update(self):
        self._blackboard.terminate_tree = True
        return py_trees.common.Status.SUCCESS