import numpy as np
from sctp.param import VEL_RATIO, RobotType, APPROX_TIME
import pytest

class Robot:
    _id_counter = 0
    def __init__(self, position, cur_node=None, at_node=True, robot_type=RobotType.Ground, \
                edge=None, iscopy =False, use_mcaction=False):
        self.id = Robot._id_counter
        Robot._id_counter += 1
        self.robot_type = robot_type
        self.cur_pose = np.array(position)
        self.last_node = cur_node
        self.unfinished_action = None
        if at_node:
            self.edge = []
            self.at_node = True
        else:
            if self.robot_type == RobotType.Ground:
                assert edge != []
                assert edge is not None, 'Edge must be provided if the UGV is not at a node'
            self.edge = edge
            self.at_node = False
            # assert self.last_node == None
        if self.robot_type == RobotType.Ground:
            self.vel = 1.0
        elif self.robot_type == RobotType.Drone:
            self.vel = 1.0*VEL_RATIO
            self.edge = []
        self.need_action = True 
        self.action = None
        self.remaining_time = 0.0
        self.direction = np.array([0.0, 0.0])
        self._cost_to_target = 0.0
        self.visited_vertices=[self.last_node]
        self.pl_vertex = self.last_node
        self.net_time = 0.0
        if not iscopy:
            self.all_poses = [[self.cur_pose[0],self.cur_pose[1]]]

    def copy(self):
        new_robot = Robot(position=self.cur_pose.copy(), cur_node=self.last_node, iscopy=True,
                          at_node=self.at_node, robot_type=self.robot_type, edge=self.edge)
        new_robot.need_action = self.need_action
        new_robot.action = self.action
        new_robot.remaining_time = self.remaining_time
        new_robot._cost_to_target = self._cost_to_target
        new_robot.vel = self.vel
        new_robot.id = self.id
        new_robot.pl_vertex = self.pl_vertex
        new_robot.direction = self.direction.copy()
        new_robot.visited_vertices = self.visited_vertices.copy()
        new_robot.unfinished_action = self.unfinished_action
        new_robot.all_poses = self.all_poses.copy()
        return new_robot
    
    def advance_time(self, delta_time):
        advance_distance = self.vel * delta_time
        self._cost_to_target -= advance_distance
        self.remaining_time -= delta_time
        if self.remaining_time < -APPROX_TIME:
            print(f'Error: Remaining time should not be negative: {self.robot_type} with ID {self.id} has remaining time of {self.remaining_time:.2f}')
        assert self.remaining_time >= -APPROX_TIME, 'Remaining time cannot be negative'
        if self.remaining_time <= APPROX_TIME:
            self.remaining_time = 0.0
            self._cost_to_target = 0.0
        
        if self.remaining_time == 0.0:
            self.need_action = True
            self.at_node = True
            self.edge = []
            if self.last_node != self.action.target:
                self.pl_vertex = self.last_node
                self.last_node = self.action.target
        elif delta_time > 0.0:
            self.at_node = False
            if self.last_node != self.action.target:
                self.edge = [self.last_node, self.action.target]
        
        self._get_coordinates_after_distance(advance_distance)
        self.net_time += delta_time


    def _get_coordinates_after_distance(self, distance):
        self.cur_pose += self.direction * distance
        self.all_poses.append([self.cur_pose[0],self.cur_pose[1]])

    def retarget(self, new_action, distance, direction):
        if not (self.remaining_time <= APPROX_TIME):
            raise NotImplementedError(f'Time remaining must be 0 for now. '
                                      f'Robot type: {self.robot_type} and ID: {self.id} and time remain: {self.remaining_time}')
        self.remaining_time = 0.0
        self.direction = direction
        self._update_time_to_target(distance)
        # Store the new action
        self.action = new_action
        self.need_action = False
    

    def _update_time_to_target(self, distance):
        self._cost_to_target = distance
        self.remaining_time = self._cost_to_target / self.vel

    def is_pose_on_edge(self, point1, point2):
        return (self.cur_pose[0]*(point1[1]-point2[1])  \
                + point1[0]*(point2[1]-self.cur_pose[1]) \
                + point2[0]*(self.cur_pose[1] - point1[1]) == 0)    


class MCRobot:
    _id_counter = 0
    def __init__(self, cur_node=None, at_node=True, robot_type=RobotType.Ground, \
                edge=None, iscopy =False):
        self.id = Robot._id_counter
        Robot._id_counter += 1
        self.robot_type = robot_type
        self.last_node = cur_node
        self.unfinished_action = None
        if at_node:
            self.edge = []
            self.at_node = True
        else:
            if self.robot_type == RobotType.Ground:
                assert edge != []
                assert edge is not None, 'Edge must be provided if the UGV is not at a node'
            self.edge = edge
            self.at_node = False
        if self.robot_type == RobotType.Ground:
            self.vel = 1.0
        elif self.robot_type == RobotType.Drone:
            self.vel = 1.0*VEL_RATIO
            self.edge = []
        self.need_action = True 
        self.action = None
        self.remaining_time = 0.0
        self._cost_to_target = 0.0
        self.net_dist = 0.0
        if not iscopy:
            self.travel_history = [self.last_node]
        self.milestones = []

    def copy(self):
        new_robot = MCRobot(cur_node=self.last_node, iscopy=True,
                          at_node=self.at_node, robot_type=self.robot_type, edge=self.edge)
        new_robot.need_action = self.need_action
        new_robot.action = self.action.copy() if self.action is not None else None
        new_robot.remaining_time = self.remaining_time
        new_robot._cost_to_target = self._cost_to_target
        new_robot.vel = self.vel
        new_robot.id = self.id
        new_robot.unfinished_action = self.unfinished_action
        new_robot.travel_history = self.travel_history.copy()
        # for maction
        # new_robot.sub_targets = self.sub_targets.copy()
        # new_robot.distances = self.distances.copy()
        new_robot.milestones = self.milestones
        return new_robot
    
    def _add_newnode_into_travel_history(self):
        assert len(self.milestones) > 0
        assert len(self.action.sub_targets) == len(self.milestones) == len(self.action.distances)
        travel_time = self.milestones[-1] / self.vel - self.remaining_time
        indices = [i for i, milestone in enumerate(self.milestones) if milestone <= travel_time]
        # print(f"{indices} vs {list(reversed(indices))} and {self.action.sub_targets}")
        visited_targets = []
        for i in list(reversed(indices)):
            if self.action.sub_targets[i] == self.last_node:
                break
            else:
                visited_targets.append(self.action.sub_targets[i])
        self.travel_history.extend(list(reversed(visited_targets)))
        return np.abs(self.milestones[-1]-travel_time) < 0.01
    
    def retarget(self, new_action):
        self.action = new_action
        self.milestones = [sum(self.action.distances[:i+1])/self.vel for i in range(len(self.action.distances))]
        self._update_time_to_target(new_action.total_dist)
        self.need_action = False

    def advance_time(self, delta_time):
        advance_distance = self.vel * delta_time
        self._cost_to_target -= advance_distance
        self.remaining_time -= delta_time
        self.at_node = self._add_newnode_into_travel_history()
        if self.remaining_time < -APPROX_TIME:
            print(f'Error: Remaining time should not be negative: {self.robot_type} with ID {self.id} has remaining time of {self.remaining_time:.2f}')
        assert self.remaining_time >= -APPROX_TIME, 'Remaining time cannot be negative'
        if self.remaining_time <= APPROX_TIME:
            self.remaining_time = 0.0
            self._cost_to_target = 0.0
        
        if len(self.travel_history) > 0:
            self.last_node = self.travel_history[-1]            
        if self.remaining_time == 0.0:
            self.need_action = True
            self.edge = []
        elif delta_time > 0.0:
            if self.at_node:
                self.edge = []
            else:
                next_node_idx = self.sub_targets.index(self.last_node) + 1
                self.edge = [self.last_node, self.sub_targets[next_node_idx]]
        self.net_dist += delta_time*self.vel


    def _update_time_to_target(self, distance):
        self._cost_to_target = distance
        self.remaining_time = self._cost_to_target / self.vel
