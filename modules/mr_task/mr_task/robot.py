import random

class Robot(object):
    _id_counter = 0
    def __init__(self, start_node):
        self.start = start_node
        self.action = None
        self.needs_action = True
        self._cost_to_target = 0
        self._start_offset = 0
        self.time_remaining = 0
        self.info_time = 0
        self.id = Robot._id_counter
        Robot._id_counter += 1
        self._same_direction = True
        self.hash_id = hash(start_node) + hash(self.id)

    def retarget(self, new_action, distances):
        if not self.time_remaining == 0:
            raise NotImplementedError('Time remaining must be 0 for now. '
                                      'Cannot switch mid-action')
        if self._cost_to_target <= 0 and self.action is not None:
            # The robot has reached to the node or passed beyond
            self.start = self.action.target_node
            self._start_offset = self._cost_to_target

        # Set the direction of the travel to know robot is traveling inside or outside start node
        self._same_direction = 1 if self.start == new_action.target_node else 0

        self._update_time_to_target(new_action, distances)

        # Store the new action
        self.action = new_action
        self.needs_action = False

    def reset_needs_action(self):
        self.needs_action = True
        self.time_remaining = 0
        self.info_time = 0
        # time_remaining and info_time are set to 0 here because retargeting wouldn't be allowed if time_remaining==0.
        # when a robot finds an object, other robot searching for the same object needs to be re-targeted.

    def advance_time(self, delta_time):
        self.info_time -= delta_time
        self.time_remaining -= delta_time
        self._cost_to_target -= delta_time
        self._start_offset = self._start_offset - self._same_direction * delta_time + \
                                                + (1 - self._same_direction) * delta_time

        self.info_time = max(0, self.info_time)
        self.time_remaining = max(0, self.time_remaining)

        # What if history says we succeed, so that we end up with negative info time?
        # There's more to consider here, I think. Will need a test to resolve this.

    def _update_time_to_target(self, new_action, distances):
        inter_node_time = distances[(self.start, new_action.target_node)]
        self._cost_to_target = self._same_direction * self._start_offset + \
                                (1 - self._same_direction) * abs(self._start_offset) + \
                                inter_node_time
        self.time_remaining = max(0, self._cost_to_target + new_action.RS)
        self.info_time = max(0, self._cost_to_target + min(new_action.RS, new_action.RE))


    def copy(self):
        new_robot = Robot(self.start)
        new_robot.action = self.action
        new_robot.needs_action = self.needs_action
        new_robot.time_remaining = self.time_remaining
        new_robot.info_time = self.info_time
        new_robot.id = self.id

        new_robot._cost_to_target = self._cost_to_target
        new_robot._start_offset = self._start_offset
        new_robot._same_direction = self._same_direction

        return new_robot

    def __eq__(self, other):
        if not isinstance(other, Robot):
            print("Robot object compared with other object")
            return False
        return self.id == other.id

    def __hash__(self):
        return self.hash_id
