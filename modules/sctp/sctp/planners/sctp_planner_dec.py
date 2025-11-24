import numpy as np
import pouct_planner
# import sctp
from sctp import gstate_dec, dstate_dec, param
from sctp.core import Action
from sctp.param import RobotType


class GroundPlanner(object):
    def __init__(self, init_graph, goalID, robot, num_rollouts=1000, C=200.0, tree_depth=25, 
                 sampling_maps=200, rollout_fn = None, verbose=False):
        self.C = C
        self.verbose = verbose
        self.observed_graph = init_graph
        self.robot = robot 
        self.goalID = goalID
        self.rollout_fn = rollout_fn
        self.goalNeighbors = []
        self.max_depth = tree_depth
        self.sampling_maps = sampling_maps
        self.num_rollouts = num_rollouts
        self.use_OptHeur = True
        
    def reached_goal(self):
        if not self.robot.at_node:
            return False
        return self.robot.last_node == self.goalID    
    
    def update(self, observations, robot_data):
        if observations:
            self.observed_graph.update(observations)
        self.robot.cur_pose = np.array([robot_data[0][0],robot_data[0][1]])
        self.robot.at_node = robot_data[1]
        self.robot.edge = robot_data[2].copy()
        self.robot.last_node = robot_data[3]
        self.robot.pl_vertex = robot_data[4]
        self.robot.remaining_time = 0.0
        self.robot.need_action = True
        
        if self.verbose:
            print('------------robots poses after updating ---------------')
            print(f"Robot: {self.robot.cur_pose} at node? {self.robot.at_node} /on edge? {self.robot.edge}/ last node: {self.robot.last_node}")
        
    
    def compute_action(self):
        if self.reached_goal():
            actions = [Action(target=self.goalID, rtype=param.Ground, start_pose=self.robot.cur_pose)]
            costs = [0.0]
            return actions, costs
        
        robot = self.robot.copy()
        sctpstate = gstate_dec.GroundState(graph=self.observed_graph, goalID=self.goalID, 
                                        robot=robot, n_maps=self.sampling_maps)
        # assert sctpstate.sampling_maps ==2
        _, _, [ordering, costs] = pouct_planner.core.po_mcts(sctpstate, \
                        n_iterations=self.num_rollouts, C=self.C, depth= self.max_depth, \
                        rollout_fn=self.rollout_fn)
        if self.verbose:
            print("Ground action ordering=", [f"{action}" for action in ordering])
        return ordering, costs
    
class DronesPlanner(object):
    def __init__(self, init_graph, goalID, drones, rollout_fn, num_rollouts=1000,
                C=200.0, sampling_maps= 200, tree_depth = 500, verbose=False):
        # self.args = args
        self.C = C
        self.verbose = verbose
        self.observed_graph = init_graph
        self.actions = []
        self.drones = drones 
        self.goalID = goalID
        self.rollout_fn = rollout_fn
        self.goalNeighbors = []
        self.robots_pos = []
        self.robots_edges = []
        self.max_depth = tree_depth
        self.num_rollouts = num_rollouts
        self.sampling_maps = sampling_maps # should we use it here or outside?
        
    def is_terminal(self):
        return all([drone.last_node == self.goalID for drone in self.drones])    
    
    def update(self, observations, drone_data, robot_data):
        if observations:
            self.observed_graph.update(observations)
        for i, drone in enumerate(self.drones):
            drone.cur_pose = np.array([drone_data[i][0][0], drone_data[i][0][1]])
            drone.at_node = drone_data[i][1]
            drone.edge = []
            drone.last_node = drone_data[i][2]
            drone.unfinished_action = drone_data[i][3]
            # drone.unfinished_action = None
            drone.remaining_time = 0.0
            drone.need_action = True    
        for i, r in enumerate(robot_data):
            self.robots_pos.append(r[0])
            self.robots_edges.append(r[2])
        if self.verbose:
            print('------------ drones poses after updating ---------------')
            print(f"Drones: ", [(drone.cur_pose, drone.at_node, drone.last_node, drone.remaining_time) for drone in self.drones])
        
    
    def compute_joint_action(self, sub_actions):
        self.actions = sub_actions
        if self.is_terminal():
            actions = [Action(target=self.goalID, rtype=RobotType.Drone, start_pose=drone.cur_pose) for drone in self.drones]
            for i, action in enumerate(actions):
                action.robotID = self.drones[i].id
            costs = [0.0 for _ in self.drones]
            return actions, costs       
        
        drones = [drone.copy() for drone in self.drones]
        for drone in drones:
            assert drone.remaining_time == 0.0
                
                
        sctpstate = dstate_dec.DronesState(actions= self.actions, robot_pos=self.robots_pos, redges=self.robots_edges,\
                                graph=self.observed_graph, restID=self.goalID, drones=drones)
        
        action, cost, [ordering, costs] = pouct_planner.core.po_mcts(sctpstate, \
                        n_iterations=self.num_rollouts, C=self.C, depth= self.max_depth, \
                        rollout_fn=self.rollout_fn)
        if len(ordering) < len(self.drones):
            ordering += [Action(target=self.goalID, rtype=RobotType.Drone) for _ in range(len(self.drones) - len(ordering))]
        if self.verbose:
            print("Drones action ordering=", [f"{action}" for action in ordering[:len(self.drones)]])
        return ordering, costs

