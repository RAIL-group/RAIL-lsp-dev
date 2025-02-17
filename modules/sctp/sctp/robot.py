import copy, random
import numpy as np
from sctp.param import VEL_RATIO, RobotType

class Robot:
    _id_counter = 0
    def __init__(self, position, cur_node, robot_type=RobotType.Ground):
        self.id = Robot._id_counter
        Robot._id_counter += 1
        self.robot_type = robot_type
        self.start = copy.copy(position)
        self.cur_pose = copy.copy(position)
        self.cur_node = cur_node # node id
        if robot_type == RobotType.Ground:
            self.vel = 1.0
        elif robot_type == RobotType.Drone:
            self.vel = 1.0*VEL_RATIO
        self.need_action = True 
        self.action = None
        self.info_time = 0.0
        self.remaining_time = 0.0
        self.direction = [1.0, 1.0]
        self._cost_to_target = 0.0

        # self.hash_id = hash(self.robot_type) + hash(self.id) + hash(self.start)


    def advance_time(self, delta_time):
        advance_distance = self.vel * delta_time
        self._cost_to_target -= advance_distance
        self.remaining_time -= delta_time
        assert self.remaining_time >= 0, 'Remaining time cannot be negative'
        self._get_coordinates_after_distance(advance_distance)


    def _get_coordinates_after_distance(self, distance):
        self.cur_pose += self.direction * distance

    def copy(self):
        new_robot = Robot(self.cur_pose, self.robot_type)
        # new_robot.net_motion = self.net_motion
        new_robot.need_action = self.need_action
        new_robot.action = self.action
        new_robot.target = self.target
        new_robot.info_time = 0.0
        new_robot.remaining_time = 0.0
        new_robot.vel = self.vel
        new_robot.id = self.id
        return new_robot

    def retarget(self, new_action, distance, direction):
        if not self.time_remaining == 0:
            raise NotImplementedError('Time remaining must be 0 for now. '
                                      'Cannot switch mid-action')
        self.direction = direction
        self._update_time_to_target(distance)

        # Store the new action
        self.action = new_action
        self.need_action = False

    def _update_time_to_target(self, distance):
        self._cost_to_target = distance
        self.time_remaining = self._cost_to_target / self.vel
        self.info_time = self.time_remaining
    
    def __hash__(self):
        return hash(self.id) + hash(self.cur_pose) + hash(self.need_action)

    
