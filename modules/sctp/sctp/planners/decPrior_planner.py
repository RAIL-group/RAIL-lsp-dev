import numpy as np
import pouct_planner
import sctp
from sctp import param
from sctp.core import Action
from sctp.param import RobotType


class SCTPPlanner(object):
    def __init__(self, init_graph, goalIDs, ugvs, uavs=[], C=200.0, rollout_num = 500, 
                 rollout_fn = None, tree_depth = 50, n_maps=100, verbose=False):
        self.rollout_num = rollout_num
        self.verbose = verbose
        self.observed_graph = init_graph
        self.ugvs = ugvs 
        self.uavs = uavs 
        self.goalIDs = goalIDs
        self.rollout_fn = rollout_fn
        self.goalNeighbors = []
        self.max_depth = tree_depth
        self.n_maps = n_maps
        self.C = C
        
    def reached_goal(self):
        return all([ugv.last_node== self.goalIDs[i] for i, ugv in enumerate(self.ugvs)])
    
    def update(self, observations, ugv_data, uav_data=None):
        if observations:
            self.observed_graph.update(observations)
        for i, ugv in enumerate(self.ugvs):
            ugv.cur_pose = np.array([ugv_data[i][0][0],ugv_data[i][0][1]])
            ugv.at_node = ugv_data[i][1]
            ugv.edge = ugv_data[i][2].copy()
            ugv.last_node = ugv_data[i][3]
            ugv.pl_vertex = ugv_data[i][4]
            ugv.remaining_time = 0.0
            ugv.need_action = True
            
        # self.robot.cur_pose = np.array([robot_data[0][0],robot_data[0][1]])
        # self.robot.at_node = robot_data[1]
        # self.robot.edge = robot_data[2].copy()
        # self.robot.last_node = robot_data[3]
        # self.robot.pl_vertex = robot_data[4]
        # self.robot.remaining_time = 0.0
        # self.robot.need_action = True
        if uav_data:
            for i, drone in enumerate(self.uavs):
                drone.cur_pose = np.array([uav_data[i][0][0], uav_data[i][0][1]])
                drone.at_node = uav_data[i][1]
                drone.edge = []
                drone.last_node = uav_data[i][2]
                drone.unfinished_action = uav_data[i][3]
                drone.remaining_time = 0.0
                drone.need_action = True    

        if self.verbose:
            print('------------These robots poses after updating ---------------')
            for i, drone in enumerate(self.uavs):
                print(f"UAV {i}: {drone.cur_pose} at node? {drone.at_node} / last node: {drone.last_node}")
            for i, ugv in enumerate(self.ugvs):
                print(f"UGV {i}: {ugv.cur_pose} at node? {ugv.at_node} /on edge? {ugv.edge}/ last node: {ugv.last_node}")
        
    
    def compute_joint_action(self):
        if self.reached_goal():
            return None, 0.0
        ugvs = [ugv.copy() for ugv in self.ugvs]
        if self.uavs == []:
            uavs = []
        else:
            uavs = [uav.copy() for uav in self.uavs]
                
        assert self.n_maps == 80
        state = sctp.dec_prior.StateDecPrior(graph=self.observed_graph, goalIDs=self.goalIDs, drones=uavs, ugvs=ugvs)
    
    
        # assert self.rollout_num == 800
        action, cost, [ordering, costs] = pouct_planner.core.po_mcts(state, \
                        n_iterations=self.rollout_num, C=self.C, depth= self.max_depth, \
                        rollout_fn=self.rollout_fn)
        
        # because replanning, so just take some first len(self.uavs)+len(self.ugvs) actions
        robot_num = len(self.uavs) + len(self.ugvs)
        # if len(ordering) < robot_num:
        #     ordering += [Action(target=self.goalID, rtype=RobotType.Drone) for _ in range(robot_num - len(ordering))]
        if self.verbose:
            print("action ordering=", [f"{action}" for action in ordering[:robot_num]])
        return ordering, costs
