import py_trees

class AlwaysRunningBehavior(py_trees.behaviour.Behaviour):
    """
    A dummy PyTrees behavior that always returns RUNNING.
    Useful for testing or keeping a behavior tree active.
    """

    def __init__(self, name="AlwaysRunningBehavior"):
        super(AlwaysRunningBehavior, self).__init__(name)

        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="is_obstacle_detected", access=py_trees.common.Access.WRITE
        )
        # self._blackboard.register_key(
        #     key="gasSensorModule_reading", access=py_trees.common.Access.WRITE
        # )

    def update(self):
        """Always returns RUNNING to keep the behavior tree active."""
        print(f"[{self.name}] 🔄 Still running...")
        self._blackboard.is_obstacle_detected = False
        # self._blackboard.gasSensorModule_reading = ["dummy_reading"]
        return py_trees.common.Status.RUNNING
