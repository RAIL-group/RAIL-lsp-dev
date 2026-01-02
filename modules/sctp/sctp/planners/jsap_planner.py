import numpy as np
import pouct_planner
import sctp
from sctp import param
from sctp.core import Action
import sctp.jsap
from sctp.param import RobotType


class JSAPPlanner(object):
    def __init__(self, init_graph, goalIDs, ugvs, uavs=[], C=200.0, rollout_num = 1000, 
                 rollout_fn = None, tree_depth = 40, n_maps=80, use_AVP = False, revisit_pen=10.0,
                 max_uanum=3, verbose=False):
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
        self.use_AVP = use_AVP
        self.max_uanum = max_uanum
        self.sampling_time = 0.0
        self.revisit_pen = revisit_pen
        self.single_policy_time = 0.0
        assert self.n_maps == 60
        
    def reached_goal(self):
        return all([ugv.last_node == self.goalIDs[i] for i, ugv in enumerate(self.ugvs)])
    
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
                print(f"UAV {i}: {drone.cur_pose} at node? {drone.at_node} /last node: {drone.last_node} to Goal: {self.goalIDs[0]}")
            for i, ugv in enumerate(self.ugvs):
                print(f"UGV {i}: {ugv.cur_pose} at node? {ugv.at_node} /on edge? {ugv.edge}/last node: {ugv.last_node} to Goal: {self.goalIDs[i]}")
        
    
    def compute_joint_action(self):
        if self.reached_goal():
            return None, 0.0
        ugvs = [ugv.copy() for ugv in self.ugvs]
        if self.uavs == []:
            uavs = []
        else:
            uavs = [uav.copy() for uav in self.uavs]
                
        assert self.n_maps == 60
        # assert self.spolicy_rollouts == 300
        # assert self.max_uanum == 1
        # assert uavs != []
        state = sctp.jsap.JSAPState(graph=self.observed_graph, goalIDs=self.goalIDs, n_maps=self.n_maps, revisit_pen=self.revisit_pen, \
                                             drones=uavs, ugvs=ugvs, useAVP=self.use_AVP, max_uanum=self.max_uanum)
    
        # assert state.uavs != []
        # assert self.rollout_num == 800
        action, cost, [ordering, costs, sampling_time, s_policy_time] = pouct_planner.core.po_mcts(state, \
                        n_iterations=self.rollout_num, C=self.C, depth= self.max_depth, \
                        rollout_fn=self.rollout_fn)
        self.sampling_time += sampling_time
        self.single_policy_time += s_policy_time
        assert self.single_policy_time == 0.0
        if self.verbose:
            print("action ordering=", [f"{action}" for action in ordering])
        return ordering, costs
