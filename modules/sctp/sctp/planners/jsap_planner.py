import numpy as np
import pouct_planner
import sctp
import sctp.jsap
import torch 
from sctp.learning.iap_gnn import load_iap_gnn_model


class JSAPPlanner(object):
    def __init__(self, init_graph, goalIDs, ugvs, uavs=[], C=200.0, rollout_num = 1000, 
                 rollout_fn = None, tree_depth = 40, n_maps=80, use_AVP = False, revisit_pen=10.0,
                 max_uanum=3, verbose=False, use_DAP=False, useLearning=False, model_path=None):
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
        self.use_DAP = use_DAP
        self.use_Learning = useLearning
        self.model_path = model_path
        self.max_uanum = max_uanum
        self.sampling_time = 0.0
        self.revisit_pen = revisit_pen
        self.single_policy_time = 0.0
        self.model = None
        self.device = None
        if self.use_Learning:
            assert model_path is not None, "Model path must be provided when use_Learning is True"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = load_iap_gnn_model(path=model_path, device=self.device)
        
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
                
        state = sctp.jsap.JSAPState(graph=self.observed_graph, goalIDs=self.goalIDs, n_maps=self.n_maps, \
                        revisit_pen=self.revisit_pen, drones=uavs, ugvs=ugvs, useAVP=self.use_AVP, \
                        useDAP=self.use_DAP, max_uanum=self.max_uanum, useLearning=self.use_Learning,\
                            gnn_model=self.model, device=self.device)
        # assert state.uavs != []
        # assert self.max_depth == 12
        # assert self.n_maps == 200
        mdepth = self.max_depth
        action, cost, [ordering, costs, sampling_time, s_policy_time] = pouct_planner.core.po_mcts(state, \
                        n_iterations=self.rollout_num, C=self.C, depth= mdepth, \
                        rollout_fn=self.rollout_fn)
        self.single_policy_time += s_policy_time
        assert self.single_policy_time == 0.0
        if self.verbose:
            print("action ordering=", [f"{action}" for action in ordering])
        return ordering, costs
