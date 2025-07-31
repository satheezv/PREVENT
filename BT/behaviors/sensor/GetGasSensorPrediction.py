import py_trees
import time
from utils.GasSensorHidInterface import HIDGasSensor

class ReadGasSensorBehavior(py_trees.behaviour.Behaviour):
    """
    A PyTrees behavior that reads gas sensor data and detects hazards.
    """

    VOC_THRESHOLD = 600 # 🚨 VOC Level at which the robot should stop

    def __init__(self, name="ReadGasSensor", sensor = None, num_reads=1):
        super(ReadGasSensorBehavior, self).__init__(name)
        self.sensor = sensor  # ✅ Initialize as None
        self.num_reads = num_reads
        self.hazard_detected = False

        self._blackboard = py_trees.blackboard.Client()

        self._blackboard.register_key(
            key="is_VOC_hazard_detected", access=py_trees.common.Access.WRITE
        )

        self._blackboard.register_key(
            key="gasSensorModule_reading", access=py_trees.common.Access.WRITE
        )


    def initialise(self):
        """Ensures the sensor connection is established only once."""
        if self.sensor is None:
            self.sensor = HIDGasSensor()
            print(f"[{self.name}] ✅ Sensor initialized at the ReadGasSensorBehavior.")
        self._blackboard.is_VOC_hazard_detected = False
        self._blackboard.gasSensorModule_reading = None

    def update(self):
        """Reads gas sensor data and detects hazards."""
        try:
            if self.sensor is None:
                print(f"[{self.name}] ❌ Sensor not initialized!")
                return py_trees.common.Status.FAILURE

            readings = self.sensor.read_multiple(num_reads=self.num_reads)
            if readings:
                for reading in readings:
                    self._blackboard.gasSensorModule_reading = reading["voc"]
                    if reading["voc"] > self.VOC_THRESHOLD:  # 🚨 VOC Hazard Detected!
                        self.hazard_detected = True
                        self._blackboard.is_VOC_hazard_detected = True
                        self._blackboard.gasSensorModule_reading = reading["voc"]
                        print(f"[{self.name}] ⚠️ Gas Hazard Detected! VOC: {reading['voc']} ppm")
                        return py_trees.common.Status.SUCCESS  # ✅ StopRobotBehavior will now trigger

                print(f"[{self.name}] ✅ No Gas Hazard detected.")
                return py_trees.common.Status.RUNNING  # ✅ Keep monitoring
            else:
                return py_trees.common.Status.FAILURE  # ❌ Sensor failure
        except Exception as e:
            print(f"[{self.name}] ❌ Sensor error: {e}")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Closes the sensor only when the entire behavior tree stops."""
        if new_status == py_trees.common.Status.INVALID and self.sensor is not None:
            self.sensor.close()
            self.sensor = None
            print(f"[{self.name}] 🔌 Sensor connection closed.")




class ReadGasSensorBehavior_Once(py_trees.behaviour.Behaviour):
    """
    A PyTrees behavior that reads gas sensor data and detects hazards.
    """

    VOC_THRESHOLD = 5000 # 🚨 VOC Level at which the robot should stop

    def __init__(self, name="ReadGasSensor", num_reads=1):
        super(ReadGasSensorBehavior_Once, self).__init__(name)
        self.sensor = HIDGasSensor()  # ✅ Initialize as None
        self.num_reads = num_reads
        self.hazard_detected = False

        self._blackboard = py_trees.blackboard.Client()

        self._blackboard.register_key(
            key="is_VOC_hazard_detected", access=py_trees.common.Access.WRITE
        )

        self._blackboard.register_key(
            key="gasSensorModule_reading", access=py_trees.common.Access.WRITE
        )

        self._blackboard.register_key(
            key="task_started", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(key="is_obstacle_detected", access=py_trees.common.Access.READ)

    def initialise(self):
        """Ensures the sensor connection is established only once."""
        if self.sensor is None:
            self.sensor = HIDGasSensor()
            print(f"[{self.name}] ✅ Sensor initialized at the ReadGasSensorBehavior.")
        self._blackboard.is_VOC_hazard_detected = False
        self._blackboard.gasSensorModule_reading = None
        self._blackboard.task_started = False

    def update(self):
        """Reads gas sensor data and detects hazards."""
        if self.name == "Mid_reading" and not self._blackboard.is_obstacle_detected:
            print(f"No obstacle detected and gas sensor read behavior {self.name} is bypassed.")
            return py_trees.common.Status.SUCCESS
        # if self.name == "Mid_reading":
        #     self._blackboard.is_VOC_hazard_detected = False
        #     self._blackboard.gasSensorModule_reading = 52.41352487252239
        #     return py_trees.common.Status.SUCCESS
        else:
            try:
                if self.sensor is None:
                    print(f"[{self.name}] ❌ Sensor not initialized!")
                    return py_trees.common.Status.SUCCESS   ## Change to failure 
                if not self.name == "Init_reading":
                    print("10s wait for the sensor to pickup the VOC")
                    time.sleep(10)
                readings = self.sensor.read_multiple(num_reads=self.num_reads)
                if readings:
                    for reading in readings:
                        self._blackboard.gasSensorModule_reading = reading["voc"]
                        if reading["voc"] > self.VOC_THRESHOLD:  # 🚨 VOC Hazard Detected!
                            self.hazard_detected = True
                            self._blackboard.is_VOC_hazard_detected = True
                            self._blackboard.gasSensorModule_reading = reading["voc"]
                            print(f"[{self.name}] ⚠️ Gas Hazard Detected! VOC: {reading['voc']}")
                            return py_trees.common.Status.SUCCESS  # ✅ StopRobotBehavior will now trigger

                    print(f"[{self.name}] ✅ No Gas Hazard detected.")
                    if self.name == "Init_reading":
                        return py_trees.common.Status.FAILURE  # ✅ Keep monitoring
                    else:
                        return py_trees.common.Status.SUCCESS  # ✅ Keep monitoring
                else:
                    return py_trees.common.Status.FAILURE  # ❌ Sensor failure
            except Exception as e:
                print(f"[{self.name}] ❌ Sensor error: {e}")
                return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Closes the sensor only when the entire behavior tree stops."""
        if new_status == py_trees.common.Status.INVALID and self.sensor is not None:
            self.sensor.close()
            self.sensor = None
            print(f"[{self.name}] 🔌 Sensor connection closed.")
