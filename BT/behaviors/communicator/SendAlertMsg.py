import py_trees
from utils.slack_msging import SlackAlertUtility



class Nav_Alert(py_trees.behaviour.Behaviour):
    def __init__(self, name, slack_alert_utility: SlackAlertUtility):

        
        """
        Behavior Tree Node for sending Slack alerts.
        :param name: Name of the behavior node
        :param slack_alert_utility: Instance of SlackAlertUtility
        :param msg_to_send: Message to send in the alert
        """
        super(Nav_Alert, self).__init__(name)

        # Black Board Variables
        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="is_VOC_hazard_detected", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="gasSensorModule_reading", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="destination_reached", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(key="is_obstacle_detected", access=py_trees.common.Access.READ)
        
        self._blackboard.register_key(key="ResNet18_best_match", access=py_trees.common.Access.READ)

        self._blackboard.register_key(
            key="last_graph_node_id", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="Emergency_stop_triggered", access=py_trees.common.Access.READ
        )

        self._blackboard.register_key(
            key="identified_hazard_infront", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="hazard_pic", access=py_trees.common.Access.READ)  
        # self.SLACK_USER_ID = "U04N321A5Q9"  #satheesh
        self.slack_alert_utility = slack_alert_utility 

    def initialise(self):
        """Called just before entering RUNNING state."""

    def update(self):
        """Action to perform during execution."""
        print("🔔 Alert behavior triggered!")  # Debugging line
        try:
            # Initialize an empty list to store messages
            messages = []
            # Check and accumulate messages
            if self._blackboard.exists("Emergency_stop_triggered") and self._blackboard.Emergency_stop_triggered:
                if self._blackboard.is_obstacle_detected and self._blackboard.is_VOC_hazard_detected and self._blackboard.identified_hazard_infront:
                    messages.append(f"Emergency stop triggered! Robot stopped due to an obstacle ({self._blackboard.ResNet18_best_match}), it is a VOC hazard with the VOC Value: {self._blackboard.gasSensorModule_reading} PPM, the hazard is identified as: {self._blackboard.identified_hazard_infront}")
                elif self._blackboard.is_obstacle_detected and self._blackboard.identified_hazard_infront:
                    messages.append(f"Emergency stop triggered! Robot stopped due to an obstacle ({self._blackboard.ResNet18_best_match}), and the identified object is {self._blackboard.identified_hazard_infront}. the voc value is {self._blackboard.gasSensorModule_reading} PPM")
                elif self._blackboard.is_VOC_hazard_detected and self._blackboard.identified_hazard_infront:
                    messages.append(f"Emergency stop triggered! Robot stopped due to a VOC hazard with the VOC Value: {self._blackboard.gasSensorModule_reading} PPM, and the identified object is {self._blackboard.identified_hazard_infront}")
            elif self._blackboard.exists("destination_reached") and self._blackboard.destination_reached:
                messages.append(f"Robotic chemist has reached the destination node: {self._blackboard.last_graph_node_id}")
            else:
                pass
            
            # Combine all messages into a single string
            final_message = "\n".join(messages) if messages else "No Messages to send"

            # Print or send the final message
            print(final_message)
            print(f"Sending alert: {final_message}")
            success = self.slack_alert_utility.send_slack_message(final_message)
            self.slack_alert_utility.send_slack_message_with_image(self._blackboard.hazard_pic)

            # success = True
            return py_trees.common.Status.SUCCESS if success else py_trees.common.Status.FAILURE
        except Exception as e:
            print(f"Failed to send alert: {e}")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Clean up when exiting the behavior."""
        pass


class Man_Alert(py_trees.behaviour.Behaviour):
    def __init__(self, name, slack_alert_utility: SlackAlertUtility):

        
        """
        Behavior Tree Node for sending Slack alerts.
        :param name: Name of the behavior node
        :param slack_alert_utility: Instance of SlackAlertUtility
        :param msg_to_send: Message to send in the alert
        """
        super(Man_Alert, self).__init__(name)

        # Black Board Variables
        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(
            key="is_VOC_hazard_detected", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="gasSensorModule_reading", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="Emergency_stop_triggered", access=py_trees.common.Access.READ
        )

        self._blackboard.register_key(
            key="task_completed", access=py_trees.common.Access.READ
        )

        self._blackboard.register_key(
            key="skill_completed", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="task_status", access=py_trees.common.Access.READ
        )

        self._blackboard.register_key(
            key="identified_hazard_infront", access=py_trees.common.Access.READ
        )
        self._blackboard.register_key(
            key="hazard_pic", access=py_trees.common.Access.READ)  
        
        self._blackboard.register_key(
            key="retract_motion_triggered", access=py_trees.common.Access.WRITE
        )
        # self.SLACK_USER_ID = "U04N321A5Q9"  #satheesh
        self.slack_alert_utility = slack_alert_utility 

    def initialise(self):
        """Called just before entering RUNNING state."""

    def update(self):
        """Action to perform during execution."""
        print("🔔 Alert behavior triggered!")  # Debugging line
        try:
            # Initialize an empty list to store messages
            messages = []
            # Check and accumulate messages
            if self._blackboard.exists("task_completed") and self._blackboard.task_completed:
                messages.append(f"Task {self._blackboard.task_status[1]}, with the Sequence id {self._blackboard.task_status[0]} is now Completed with status {self._blackboard.task_status[2]}(0. Executing, 1. Waiting, 2. Finished, 3. Error ).")
                self._blackboard.skill_completed = True
            else:
                messages.append(f"Hazard detected - type : {self._blackboard.identified_hazard_infront}, with VOC reading : {self._blackboard.gasSensorModule_reading} ")
            
            # Combine all messages into a single string
            final_message = "\n".join(messages) if messages else "No Messages to send"

            # Print or send the final message
            print(final_message)
            print(f"Sending alert: {final_message}")
            success = self.slack_alert_utility.send_slack_message(final_message)
            if self._blackboard.exists("hazard_pic"):
                self.slack_alert_utility.send_slack_message_with_image(self._blackboard.hazard_pic)

            # success = True
            return py_trees.common.Status.SUCCESS # if success else py_trees.common.Status.FAILURE
        except Exception as e:
            print(f"Failed to send alert: {e}")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Clean up when exiting the behavior."""
        pass

