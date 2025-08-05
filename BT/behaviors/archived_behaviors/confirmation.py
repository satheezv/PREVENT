import py_trees
from .terminate_tree import TerminateTree

class ConfirmHazard(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ConfirmHazard, self).__init__(name)
    def update(self):
        hazard_confirmed = self.check_gas_sensor()
        if hazard_confirmed:
            print("Hazard confirmed by gas sensor.")
            return py_trees.common.Status.SUCCESS
        else:
            print("No hazard confirmed by gas sensor.")
            return py_trees.common.Status.FAILURE
    def check_gas_sensor(self):
        return True  # Simulated confirmation