from sctp import graph as g
from sctp.utils import paths, plotting
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from sctp import param, core
from sctp import action_estimation as ae
from sctp.learning.iap_gnn import load_iap_gnn_model
from sctp.utils import underlying_graph as ug
import time
import torch

class JSAPState(object):
    total_sampling_time = 0.0
    @classmethod
    def reset_sampling_time(cls):
        cls.total_sampling_time = 0.0
    def __init__(self, graph=None, goalIDs=[], ugvs=[], drones=[], iscopy=False, n_maps=60, 
                 useAVP=False, useDAP=False, useLearning=False, max_uanum=3, revisit_pen=20.0,
                 gnn_model=None, device=None, gnn_cache=None):
        self.action_cost = 0.0
        self.heuristic = -1.0
        self.depth = 0
        self.vertices_map = dict() # map vertex id to vertex object
        self.sampling_maps = n_maps
        self.state_actions = []
        self.use_OptiHeur = True
        self.noway2goal = False
        self.cur_ugv_idx = -1
        self.avail_uav_actions = []
        
        self.sampling_time = 0.0 #measure the time for sampling-maps
        self.s_policy_time = 0.0 # measure the time for single policy computation
        self.ugvs_time = 0.0
        self.uavs_time = 0.0
        self.revisit_pen = revisit_pen
        self.action_values = dict() # map action to its value
        self.behavior_change = dict() # map action to its value
        self.got_sampling_time = False
        self.gnn_cache = gnn_cache # cache for GNN predictions, map from state representation to action values
        
        if not iscopy: # the first state
            # need to filter the visited POIs
            assert graph is not None
            assert ugvs != []
            self.graph = graph
            self.max_uanum = max_uanum
            self.goalIDs = goalIDs
            self.history = core.History()
            self.vertices_map = {v.id: v for v in self.graph.vertices + self.graph.pois}
            self.v_vertices = dict()
            self.assigned_pois = set()
            self.init_history()
            self.ugvs = ugvs
            self.use_AVP = useAVP
            self.use_DAP = useDAP
            self.use_Learning = useLearning
            self.device = device
            self.model = None
            # set up underlying graph for sampling
            self.edges = ug.get_initial_edges(self.graph)
            self.pg_positions = ug.get_vertex_positions(self.graph.vertices)
            self.pg_adjacency, self.pg_probabilities = ug.create_adj_prob_matrices(self.edges, self.pg_positions)
            # set the pruning techniques
            if self.use_AVP or self.use_DAP:
                assert self.use_AVP != self.use_DAP
                assert self.use_Learning == False, "AVP and DAP are not compatible with Learning-based pruning"
            if self.use_Learning:
                assert self.use_AVP==False and self.use_DAP==False, "Learning-based pruning is not compatible with AVP or DAP"
                # assert model_path is not None, "Model path must be provided when using learning-based pruning"
                # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = gnn_model # load_iap_gnn_model(path=model_path, device=self.device)
            self.ugvs_actions = [[] for _ in range(len(self.ugvs))]
            for i, ugv in enumerate(self.ugvs):
                if ugv.last_node == self.goalIDs[i]:
                    ugv.need_action = True
                    ugv_actions = [core.Action(target=self.goalIDs[i], start_pose=(ugv.cur_pose[0],ugv.cur_pose[1]))]
                    ugv_actions[0].update_robotID(i)
                else:
                    ugv.need_action = True
                    ugv_actions = get_ugv_action(self, i)
                    if ugv_actions == []:
                        self.noway2goal = True
                        self.action_cost = param.STUCK_COST
                ugv.visited_vertices.append(ugv.last_node)                
                # assert len(ugv_actions) > 0
                self.ugvs_actions[i] = ugv_actions
            self.cur_ugv_idx = 0
            idx = [i for i, robot in enumerate(self.ugvs) if robot.need_action==True]
            assert len(idx) > 0
            self.state_actions = [action for action in self.ugvs_actions[idx[0]]] # what if this ugv has reached goal?
            self.update_heuristic()
            
            self.uavs = drones
            if len(self.uavs) > 0:
                self.uav_actions = [core.Action(target=poi.id, rtype=param.RobotType.Drone) for poi in self.graph.pois]                
                self.uav_actions = [action for action in self.uav_actions \
                                    if self.history.get_action_outcome(action) == param.EventOutcome.CHANCE] # list unexplored pois
                for i, uav in enumerate(self.uavs):
                    if uav.unfinished_action and uav.unfinished_action in self.uav_actions:
                        uav.action = uav.unfinished_action
                        assert uav.action.rtype == param.RobotType.Drone
                        uav.unfinished_action = None
                        uav.need_action = False
                        self.assigned_pois.add(uav.action.target)
                        uav.action.update_pose((uav.cur_pose[0], uav.cur_pose[1]))
                        uav.action.update_robotID(i)
                        distance, direction = self.get_distance_direction(uav.cur_pose, uav.action.target)
                        uav.retarget(uav.action, distance, direction)
                        self.uav_actions.remove(uav.action)
                    else:
                        uav.need_action = True
                        uav.action = None
                
                indices = [i for i, uav in enumerate(self.uavs) if uav.need_action]
                time1 = time.perf_counter()
                if self.use_AVP:
                    self.avail_uav_actions = [core.Action(target=act.target, rtype=param.RobotType.Drone) for act in self.uav_actions]
                    self.update_action_bc_networkX()
                    if len(self.avail_uav_actions) != len(self.behavior_change):
                        raise ValueError("The length of avail_uav_actions and behavior_change do not match in __init__()")
                    if indices != []:
                        self.uav_actions = ae.get_uav_action_2ag(self, indices[0])
                        self.sampling_time += (time.perf_counter() - time1)
                        JSAPState.total_sampling_time += (time.perf_counter() - time1)
                        assert len(self.uav_actions) <= self.max_uanum
                if self.use_DAP:
                    self.avail_uav_actions = [core.Action(target=act.target, rtype=param.RobotType.Drone) for act in self.uav_actions]
                    if len(indices) > 0:
                        self.uav_actions = ae.get_closest_actions(self, indices[0])
                        JSAPState.total_sampling_time += (time.perf_counter() - time1)
                        assert len(self.uav_actions) <= self.max_uanum
                
                if self.use_Learning: # using learning
                    self.avail_uav_actions = [core.Action(target=act.target, rtype=param.RobotType.Drone) for act in self.uav_actions]
                    if len(indices) > 0:
                        self.uav_actions = ae.get_uav_action_gnn(self, indices[0])
                        JSAPState.total_sampling_time += (time.perf_counter() - time1)
                         
                if len(self.uav_actions) == 0:                    
                    self.uav_actions = [core.Action(target=self.goalIDs[0], rtype=param.RobotType.Drone)]
                    if indices != []:
                        self.uav_actions[0].update_pose((self.uavs[indices[0]].cur_pose[0], self.uavs[indices[0]].cur_pose[1]))
                        self.uav_actions[0].update_robotID(indices[0])      
                # check right here
                assert isinstance(self.uav_actions[0], core.Action)
                assert len(self.uav_actions) > 0
                if any([uav.need_action for uav in self.uavs]):
                    self.cur_ugv_idx = -1
                    self.state_actions = [action for action in self.uav_actions]
                
    def get_actions(self):
        return self.state_actions
    
    def init_history(self):
        for vertex in self.graph.vertices+self.graph.pois:
            action = core.Action(target=vertex.id)
            if vertex.block_prob == 1.0:
                self.history.add_history(action, param.EventOutcome.BLOCK)
            elif vertex.block_prob == 0.0:
                self.history.add_history(action, param.EventOutcome.TRAV)
                

    def update_heuristic(self):
        block_pois = [key.target for key, value in self.history.get_data().items() if value == core.EventOutcome.BLOCK]
        edges, vertices_wrobot = g.get_removedEdges_allrobots(self.graph, self.ugvs, block_pois)
        new_pois = [poi for poi in block_pois if poi not in vertices_wrobot]        
        
        new_graph = g.remove_pois(self.graph, new_pois)
        new_graph = g.remove_edges(new_graph, edges)
        self.heuristic = 0.0
        assert self.use_OptiHeur == True
        for i, robot in enumerate(self.ugvs):
            heuristic_cost = 0.0
            if robot.at_node:
                redge = [robot.last_node, robot.last_node]
                if self.use_OptiHeur:
                    heuristic_cost, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[0], goal=self.goalIDs[i])
                    if heuristic_cost < 0.0:
                        self.noway2goal = True
                        self.action_cost = param.STUCK_COST
                else:
                    assert 1 == 0
                    heuristic_cost = core.sampling_rollout(new_graph, [robot.last_node,robot.last_node], 0.0, 0.0, self.goalIDs[i], 
                                                robot.at_node, startNode=robot.last_node, n_samples=self.n_maps)
            else:
                redge = [robot.edge[0], robot.edge[1]]            
                d1 = np.linalg.norm(np.array(robot.cur_pose)-np.array(self.vertices_map[redge[0]].coord))
                d2 = np.linalg.norm(np.array(robot.cur_pose)-np.array(self.vertices_map[redge[1]].coord))
                if self.use_OptiHeur:
                    min_dist1, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[0], goal=self.goalIDs[i])
                    min_dist2, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[1], goal=self.goalIDs[i])
                    if (min_dist1 < 0) != (min_dist2 < 0):
                        new_graph.print_graph_config()
                        if min_dist1 < 0:
                            print(f"UGV {i} is on edge: ({redge}) and Vertex {redge[0]} to goal {self.goalIDs[i]} is blocked")
                        if min_dist2 < 0:
                            print(f"UGV {i} is on edge: ({redge}) and Vertex {redge[1]} to goal {self.goalIDs[i]} is blocked")
                        print(f"Other robots positions: ")
                        for j, other_robot in enumerate(self.ugvs):
                            if j != i:
                                if other_robot.at_node:
                                    print(f"UGV {j} is at node {other_robot.last_node} at position {other_robot.cur_pose}")
                                else:
                                    print(f"UGV {j} on edge ({other_robot.edge}) at position {other_robot.cur_pose}")
                        raise ValueError("One path is blocked but the other is not, error in shortest path computation")
                        
                    # assert (min_dist1 < 0) == (min_dist2 < 0)
                    if min_dist1 < 0.0 and min_dist2 < 0.0:
                        heuristic_cost = -1.0
                        self.noway2goal = True
                        self.action_cost = param.STUCK_COST
                    else:
                        heuristic_cost = min(d1 + min_dist1, d2 + min_dist2)
                else:
                    assert 1 == 0
                    heuristic_cost = core.sampling_rollout(new_graph, redge, d1, d2, self.goalIDs[i], robot.at_node,
                                                startNode=robot.last_node, n_samples=self.n_maps)
            heuristic_cost = param.NOWAY_PEN if heuristic_cost < 0.0 else heuristic_cost
            self.heuristic += heuristic_cost                   
        return self.heuristic
    
    def update_action_bc(self):
        assert self.use_DAP == False
        self.behavior_change.clear()
        self.action_values.clear()
        time1 = time.perf_counter()
        min_bc = 0.0
        for ii, act in enumerate(self.avail_uav_actions):
            if act.target in self.assigned_pois:
                continue
            if self.graph.get_poi(act.target).block_prob < 0.05 or self.graph.get_poi(act.target).block_prob > 0.95:
                bc_value = 0.0
            else:
                bc_value = ae.get_ugvs_behavior_change(state=self, action=act)
            if bc_value < min_bc:
                min_bc = bc_value
            self.behavior_change[act] = bc_value
        if min_bc < 0.0:
            for key in self.behavior_change.keys():
                self.behavior_change[key] -= min_bc
        JSAPState.total_sampling_time += time.perf_counter() - time1
    
    def update_action_bc_networkX(self):
        assert self.use_DAP == False
        assert self.use_Learning == False
        self.behavior_change.clear()
        self.action_values.clear()
        time1 = time.perf_counter()
        pg = ug.ProbabilisticGraph(positions=self.pg_positions, adjacency=self.pg_adjacency, \
                                        probabilities=self.pg_probabilities)
        min_bc = 0.0
        
        for ii, act in enumerate(self.avail_uav_actions):
            if act.target in self.assigned_pois:
                continue
            if self.graph.get_poi(act.target).block_prob < 0.05 or self.graph.get_poi(act.target).block_prob > 0.95:
                bc_value = 0.0
            else:
                bc_value = ae.get_ugvs_bc_networkX(state=self, action=act, pg=pg)
            if bc_value < min_bc:
                min_bc = bc_value
            self.behavior_change[act] = bc_value
        if min_bc < 0.0:
            for key in self.behavior_change.keys():
                self.behavior_change[key] -= min_bc
        JSAPState.total_sampling_time += time.perf_counter() - time1
        
    @property
    def is_goal_state(self):
        return all ([ugv.last_node == self.goalIDs[i] for i, ugv in enumerate(self.ugvs)]) or self.noway2goal

    @property
    def is_block_state(self):
        return self.noway2goal

    def copy(self):
        new_state = JSAPState(iscopy=True, gnn_cache=self.gnn_cache)
        new_state.cur_ugv_idx = self.cur_ugv_idx
        new_state.heuristic = self.heuristic
        new_state.vertices_map = self.vertices_map.copy()
        new_state.v_vertices = self.v_vertices.copy()
        new_state.depth = self.depth
        new_state.graph = self.graph
        new_state.sampling_maps = self.sampling_maps
        if new_state.depth >= 3:
            new_state.sampling_maps = max(25, new_state.sampling_maps - 10*(new_state.depth -3))
        new_state.goalIDs = self.goalIDs.copy()
        new_state.use_OptiHeur = self.use_OptiHeur
        new_state.use_AVP = self.use_AVP
        new_state.use_DAP = self.use_DAP
        new_state.use_Learning = self.use_Learning
        new_state.device = self.device
        new_state.model = self.model
        if new_state.use_AVP or new_state.use_DAP:
            assert new_state.use_AVP != new_state.use_DAP
        new_state.max_uanum = self.max_uanum
        # assert new_state.max_uanum == 5
        new_state.action_cost = 0.0
        new_state.revisit_pen = self.revisit_pen
        new_state.assigned_pois = self.assigned_pois.copy() # [poi for poi in self.assigned_pois]
        new_state.history = self.history.copy()
        # save the underlying graph
        new_state.pg_positions = self.pg_positions
        new_state.pg_adjacency = self.pg_adjacency
        new_state.pg_probabilities = self.pg_probabilities.copy()
        new_state.edges = self.edges
        # save the pruning techniques
        if self.use_AVP:
            new_state.avail_uav_actions = self.avail_uav_actions.copy()
            new_state.behavior_change = self.behavior_change.copy()
            new_state.action_values = self.action_values.copy()
            if len(new_state.avail_uav_actions) != len(new_state.behavior_change):
                raise ValueError("Debugging of action numbers in copy()")
        elif self.use_DAP:
            new_state.avail_uav_actions = self.avail_uav_actions.copy()
        elif self.use_Learning:
            new_state.avail_uav_actions = self.avail_uav_actions.copy()
        else:
            new_state.behavior_change = None
            new_state.action_values = None
            new_state.avail_uav_actions = None
            
        new_state.state_actions = []
        # copy the robot
        new_state.ugvs = [ugv.copy() for ugv in self.ugvs]
        new_state.ugvs_actions = [[core.Action(target=action.target, start_pose=(action.start_pose[0],action.start_pose[1])) \
                                    for action in ugv_actions] for ugv_actions in self.ugvs_actions]
        for ii, actions in enumerate(new_state.ugvs_actions):
            for act in actions:
                act.update_robotID(ii)
        if self.uavs != []:
            new_state.uavs = [uav.copy() for uav in self.uavs]
            new_state.uav_actions = [core.Action(target=action.target, rtype=param.RobotType.Drone) \
                                    for action in self.uav_actions]
        else:
            new_state.uavs = []
            new_state.uav_actions = []
            
        return new_state
        
    def transition(self, action):
        temp_state = self.copy()
        if action.rtype == param.RobotType.Drone:
            uav_needs_action = [uav.need_action for uav in temp_state.uavs]
            assert any(uav_needs_action) == True
            
            if self.use_AVP or self.use_DAP or self.use_Learning:
                uav_idx = action.robotID
            else:
                uav_idx = uav_needs_action.index(True)
                action.update_robotID(uav_idx)
            
            if isinstance(uav_idx, int) == False:
                print(f"The depth of the state is {temp_state.depth}")
                raise ValueError(f"uav_idx is not an integer: {uav_idx}")
            start_pos = (temp_state.uavs[uav_idx].cur_pose[0], temp_state.uavs[uav_idx].cur_pose[1])
            action.update_pose(start_pos)
            
            assert action in temp_state.uav_actions
            if temp_state.cur_ugv_idx != -1:
                # print(f"Assigning action {action.target} to drone {uav_idx} when cur_ugv_idx is {temp_state.cur_ugv_idx}")
                raise ValueError("ERROR: cur_ugv_idx should be -1 when assigning action to UAV")
            if action.target in temp_state.assigned_pois:
                # print(f"Assigning action {action.target} to drone {uav_idx} that is in the assigned_pois {temp_state.assigned_pois}")
                raise ValueError("Assigned POI is already assigned to other UAV")
            if np.isnan(start_pos[0]) or np.isnan(start_pos[1]):
                ValueError("Start position is NaN") 
            distance, direction = temp_state.get_distance_direction(start_pos, action.target)            
            temp_state.uavs[uav_idx].retarget(action, distance, direction)
            if action.target != temp_state.goalIDs[0]:
                temp_state.assigned_pois.add(action.target)
                if self.use_AVP:
                    assert len(temp_state.avail_uav_actions) == len(temp_state.behavior_change)
                    assert len(temp_state.avail_uav_actions) == len(temp_state.action_values)
                    num_elements_before = len(temp_state.avail_uav_actions)
                    temp_state.avail_uav_actions.remove(action)
                    temp_state.behavior_change.pop(action, None)
                    temp_state.action_values.pop(action, None)
                    assert len(temp_state.avail_uav_actions) == num_elements_before - 1
                    assert len(temp_state.behavior_change) == len(temp_state.avail_uav_actions)
                elif self.use_DAP or self.use_Learning:
                    temp_state.avail_uav_actions.remove(action)
                else:
                    temp_state.uav_actions.remove(action)
                
        elif action.rtype == param.RobotType.Ground:
            assert temp_state.cur_ugv_idx > -1
            ugv_needs_action = [ugv.need_action for ugv in temp_state.ugvs]
            assert any(ugv_needs_action) == True
            ugv_idx = action.robotID
            start_pos = (temp_state.ugvs[ugv_idx].cur_pose[0], temp_state.ugvs[ugv_idx].cur_pose[1])
            action.update_pose(start_pos)
            action.update_robotID(ugv_idx)
            distance, direction = temp_state.get_distance_direction(start_pos, action.target)
            temp_state.ugvs[ugv_idx].retarget(action, distance, direction)
        else:
            raise ValueError("Unknown robot type in action in transition function")
        
        return advance_state(temp_state, action)

    def get_distance_direction(self, start_pos, target):
        end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == target][0].coord
        distance = np.linalg.norm(np.array(start_pos) - np.array(end_pos))
        if distance != 0.0:
            direction = (np.array([end_pos[0], end_pos[1]]) - start_pos)/distance
        else:
            direction = np.array([1.0, 1.0])
        return distance, direction

def get_ugv_action(state, ugv_idx):
    
    actions = []
    ugv = state.ugvs[ugv_idx]
    if ugv.at_node:
        if ugv.last_node == state.goalIDs[ugv_idx]:
            actions = [core.Action(target=state.goalIDs[ugv_idx], start_pose=(ugv.cur_pose[0],ugv.cur_pose[1]))]
        else:
            if state.history.get_action_outcome(core.Action(target=ugv.last_node))==param.EventOutcome.BLOCK:
                actions = [core.Action(target=ugv.pl_vertex, start_pose=(ugv.cur_pose[0],ugv.cur_pose[1]))]
            else:
                neighbors = [node for node in state.graph.vertices+state.graph.pois if node.id == ugv.last_node][0].neighbors
                actions = [core.Action(target=neighbor, start_pose=(ugv.cur_pose[0],ugv.cur_pose[1])) for neighbor in neighbors]
        ugv.visited_vertices.append(ugv.last_node)
    else:
        actions = [core.Action(target=ugv.edge[0], start_pose=(ugv.cur_pose[0],ugv.cur_pose[1])), 
                              core.Action(target=ugv.edge[1],start_pose=(ugv.cur_pose[0],ugv.cur_pose[1]))]
    actions = [action for action in actions \
                    if state.history.get_action_outcome(action) != param.EventOutcome.BLOCK]
    if len(actions) >= 2:
        actions = [action for action in actions if action.target != ugv.pl_vertex]
    [action.update_robotID(ugv_idx) for action in actions]
    return actions

def advance_state(state, action):
    # 1. if any robot needs action, determine its actions then return
    uavs_need_action = [uav.need_action for uav in state.uavs]
    if state.uavs != [] and any(uavs_need_action): # and action.rtype == param.RobotType.Drone:
        uav_idx = uavs_need_action.index(True)
        time1 = time.perf_counter()
        if state.use_AVP:
            state.uav_actions = ae.get_uav_action_2ag(state, uav_idx)
        elif state.use_DAP:
            state.uav_actions = ae.get_closest_actions(state, uav_idx)
            JSAPState.total_sampling_time += time.perf_counter() - time1
        elif state.use_Learning:
            state.uav_actions = ae.get_uav_action_gnn(state, uav_idx)
            JSAPState.total_sampling_time += time.perf_counter() - time1
        if len(state.uav_actions) ==0:
            rest_action = core.Action(target=state.goalIDs[0], rtype=param.RobotType.Drone)
            rest_action.update_pose((state.uavs[uav_idx].cur_pose[0], state.uavs[uav_idx].cur_pose[1]))
            rest_action.update_robotID(uav_idx)
            state.uav_actions = [rest_action]
        
        state.state_actions = [action for action in state.uav_actions]
        state.cur_ugv_idx = -1
        if len(state.state_actions) == 0:
            for robot in state.uavs+state.ugvs:
                print(f"{robot.robot_type} {robot.id} need_action: {robot.need_action} is atNode?? {robot.at_node} at node {robot.last_node} at {robot.cur_pose} from {robot.pl_vertex}" )
            ValueError("No available action for UAVs")
        state.depth += 1
        
        
        return {state: (1.0, 0.0)}
    robots_need_action = [robot.need_action for robot in state.ugvs]
    if any(robots_need_action):
        # set action for this state        
        robot_idx = robots_need_action.index(True)
        state.ugvs_actions[robot_idx] = [action for action in state.ugvs_actions[robot_idx] \
                            if state.history.get_action_outcome(action) != param.EventOutcome.BLOCK]
        if len(state.ugvs_actions[robot_idx]) == 0:
            state.noway2goal = True
            state.action_cost = param.STUCK_COST
            state.state_actions = []
        else:
            assert all ([action.robotID == robot_idx for action in state.ugvs_actions[robot_idx]])
            assert len(state.ugvs_actions[robot_idx]) > 0
        state.state_actions = [action for action in state.ugvs_actions[robot_idx]]
        assert all ([action.robotID == robot_idx for action in state.state_actions])
        state.cur_ugv_idx = robot_idx
        state.depth += 1
        
        return {state: (1.0, 0.0)}
    
    # 2. Find the robot that finishes its action first.
    robot_reach_first, rd_index, time_advance = _get_robot_that_finishes_first(state)
    assert time_advance >= 0.0
    state.action_cost = time_advance
    # save some data before moving
    last_nodes = [robot.last_node for robot in state.ugvs]
    edges = [robot.edge for robot in state.ugvs]
    # move the robots
    for i, robot in enumerate(state.ugvs):
        if robot.last_node != state.goalIDs[i]:
            robot.advance_time(time_advance)
        else:
            robot.need_action = False
    for uav in state.uavs:
        if uav.last_node != state.goalIDs[0]:
            uav.advance_time(time_advance)
        else:
            uav.need_action = False
        
    if robot_reach_first:
        return get_ugv_belief(state, last_nodes=last_nodes, robot_idx = rd_index, last_edges=edges)
    else:
        return get_uav_belief(state=state, uav_index=rd_index)
    

def get_ugv_belief(state, last_nodes, robot_idx, last_edges): # need to work on this.
    vertex_status = state.history.get_action_outcome(state.ugvs[robot_idx].action)
    vertex = [node for node in state.graph.vertices+state.graph.pois if node.id == state.ugvs[robot_idx].action.target][0]
    state.ugvs[robot_idx].visited_vertices.append(state.ugvs[robot_idx].last_node)
    state.v_vertices[state.ugvs[robot_idx].last_node] = state.v_vertices.get(state.ugvs[robot_idx].last_node, 0) + 1
    assert state.ugvs[robot_idx].at_node == True
    state.cur_ugv_idx = robot_idx
    # One UGV is processed one a time
    for i, ugv in enumerate(state.ugvs):
        if i != robot_idx:
            ugv.need_action = False
    if vertex_status == param.EventOutcome.BLOCK:
        state.ugvs_actions[robot_idx] = [core.Action(target=state.ugvs[robot_idx].pl_vertex, \
                            start_pose=(state.ugvs[robot_idx].cur_pose[0],state.ugvs[robot_idx].cur_pose[1]))]
        for action in state.ugvs_actions[robot_idx]:
            action.update_robotID(robot_idx)
        state.state_actions = [action for action in state.ugvs_actions[robot_idx]]
        state.update_heuristic()
        state.depth += 1
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.TRAV: 
        if state.ugvs[robot_idx].last_node == state.goalIDs[robot_idx]:
            state.ugvs_actions[robot_idx] = [core.Action(target=state.goalIDs[robot_idx], \
                        start_pose=(state.ugvs[robot_idx].cur_pose[0],state.ugvs[robot_idx].cur_pose[1]))]
        else:
            neighbors = [nei for nei in state.vertices_map[state.ugvs[robot_idx].last_node].neighbors]
            if (state.ugvs[robot_idx].pl_vertex != state.ugvs[robot_idx].last_node) and state.ugvs[robot_idx].pl_vertex in neighbors:
                neighbors.remove(state.ugvs[robot_idx].pl_vertex)
            if len(neighbors) > 1:
                state.ugvs_actions[robot_idx] = get_ugv_action(state, robot_idx)
                if state.ugvs_actions[robot_idx] == []:
                    state.noway2goal = True
                    state.action_cost = param.STUCK_COST
            elif len(neighbors) == 1:
                state.ugvs_actions[robot_idx] = [core.Action(target=neighbors[0], \
                    start_pose=(state.ugvs[robot_idx].cur_pose[0],state.ugvs[robot_idx].cur_pose[1]))]
            else:
                state.noway2goal = True
                state.action_cost = param.STUCK_COST
                state.ugvs_actions[robot_idx] = []
        assert isinstance(robot_idx, int)
        for action in state.ugvs_actions[robot_idx]:
            action.update_robotID(robot_idx)
            assert action.robotID == robot_idx
        if len(state.ugvs)==1 and len(state.uavs)==0:
            state.action_cost += (state.v_vertices.get(state.ugvs[robot_idx].last_node, 0)-1) * state.revisit_pen
        state.state_actions = [action for action in state.ugvs_actions[robot_idx] ]
        state.update_heuristic()
        state.depth += 1
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.CHANCE:
        if len(state.uavs) > 0:
            reset_uavs_action(state, robot_idx)
        
        if state.use_AVP:
            state.avail_uav_actions = [act for act in state.avail_uav_actions if act.target != state.ugvs[robot_idx].last_node]
            state.behavior_change.pop(core.Action(target=state.ugvs[robot_idx].last_node), None)
            state.action_values.pop(core.Action(target=state.ugvs[robot_idx].last_node), None)
        elif state.use_DAP or state.use_Learning:
            state.avail_uav_actions = [act for act in state.avail_uav_actions if act.target != state.ugvs[robot_idx].last_node]
        else:
            state.uav_actions = [act for act in state.uav_actions if act.target != state.ugvs[robot_idx].last_node]
        # TRAVERSABLE
        new_state_trav = get_new_ugv_node(state, robot_idx=robot_idx)
        new_state_block = get_new_ugv_node(state, robot_idx=robot_idx, last_node=last_nodes[robot_idx], blocked=True)
        
        return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                    new_state_block: (vertex.block_prob, new_state_block.action_cost)}
        
def get_new_ugv_node(state, robot_idx, last_node=None, blocked=False):
    new_state = state.copy()
    new_state.cur_ugv_idx = robot_idx
    new_state.action_cost = state.action_cost
    
    if blocked:
        prob = 1.0
        new_state.history.add_history(state.ugvs[robot_idx].action, param.EventOutcome.BLOCK)
        assert last_node is not None    
        if last_node != new_state.ugvs[robot_idx].last_node:
            target = last_node
        else:
            target = new_state.ugvs[robot_idx].pl_vertex
        new_state.ugvs_actions[robot_idx] = [core.Action(target=target, \
                    start_pose=(state.ugvs[robot_idx].cur_pose[0], state.ugvs[robot_idx].cur_pose[1]))]
        edge = state.graph.get_poi(new_state.ugvs[robot_idx].last_node).neighbors
        edge = [edge[0]-1, edge[1]-1]
    else:
        prob = 0.0
        new_state.history.add_history(state.ugvs[robot_idx].action, param.EventOutcome.TRAV)
        neighbors = [node for node in state.graph.vertices+state.graph.pois if node.id == state.ugvs[robot_idx].last_node][0].neighbors
        new_state.ugvs_actions[robot_idx] = [core.Action(target=neighbor, \
                                    start_pose=(state.ugvs[robot_idx].cur_pose[0],state.ugvs[robot_idx].cur_pose[1])) \
                                    for neighbor in neighbors if neighbor != state.ugvs[robot_idx].pl_vertex]
        edge = [neighbors[0]-1, neighbors[1]-1]
    new_state.pg_probabilities = ug.set_edge_probabilities(probs=np.array([prob]), edges=[edge], probabilities=new_state.pg_probabilities)
    if len(new_state.ugvs_actions[robot_idx]) == 0:
        new_state.noway2goal = True
        new_state.action_cost = param.STUCK_COST
    for action in new_state.ugvs_actions[robot_idx]:
        action.update_robotID(robot_idx)
            
    for i, robot in enumerate(new_state.ugvs): # reset all other UGVs if they are in the middle of their action
        if i != robot_idx and not robot.at_node: 
            robot.need_action = True
            robot.remaining_time = 0.0
            new_state.ugvs_actions[i] = get_ugv_action(new_state, i)
            if new_state.ugvs_actions[i] == []:
                new_state.noway2goal = True
                new_state.action_cost = param.STUCK_COST
            
    new_state.depth += 1
    new_state.update_heuristic()
    uav_needs_action = [i for i, uav in enumerate(new_state.uavs) if uav.need_action==True]
    if len(uav_needs_action)>0:
        time1 = time.perf_counter()
        if new_state.use_AVP:
            new_state.update_action_bc_networkX()
            new_state.uav_actions = ae.get_uav_action_2ag(new_state, uav_needs_action[0])
        elif new_state.use_DAP:
            new_state.uav_actions = ae.get_closest_actions(new_state, uav_needs_action[0])
            JSAPState.total_sampling_time += time.perf_counter() - time1
        elif new_state.use_Learning:
            new_state.uav_actions = ae.get_uav_action_gnn(state, uav_needs_action[0])
            JSAPState.total_sampling_time += time.perf_counter() - time1
        new_state.state_actions = [action for action in new_state.uav_actions]
        new_state.cur_ugv_idx = -1
    else:
        new_state.state_actions = [action for action in new_state.ugvs_actions[robot_idx]]    
    return new_state

def get_uav_belief(state, uav_index):
    assert len(state.uavs) > 0
    vertex_status = state.history.get_action_outcome(state.uavs[uav_index].action)
    vertex = [node for node in state.graph.pois+state.graph.vertices if node.id == state.uavs[uav_index].action.target][0]
    # determine all actions related to the poi and remove it from the set.
    poi_id = state.uavs[uav_index].last_node
    assert poi_id == vertex.id
    # if some uavs also finish theirs actions, reset need_action for the next iteration
    for i, uav in enumerate(state.uavs):
        uav.need_action = False if i != uav_index and uav.need_action else uav.need_action
    # just assuming ugvs do not need actions now
    for i, robot in enumerate(state.ugvs):
        robot.need_action = False
    state.cur_ugv_idx = -1
    if state.use_AVP:
        state.avail_uav_actions = [act for act in state.avail_uav_actions if act.target != poi_id]
        state.behavior_change.pop(core.Action(target=poi_id), None)
        state.action_values.pop(core.Action(target=poi_id), None)
        if len(state.avail_uav_actions) != len(state.behavior_change):
            print(f"Length of avail_uav_actions {len(state.avail_uav_actions)} and behavior_change {len(state.behavior_change)} do not match!")
            raise ValueError("Debugging of action numbers in get_uav_belief()")
    elif state.use_DAP or state.use_Learning:
        state.avail_uav_actions = [act for act in state.avail_uav_actions if act.target != poi_id]
    else:
        state.uav_actions = [act for act in state.uav_actions if act.target != poi_id]
    if vertex_status == param.EventOutcome.BLOCK: # should not go here
        state.depth += 1
        time1 = time.perf_counter()
        if state.use_AVP:
            state.update_action_bc_networkX()
            state.uav_actions = ae.get_uav_action_2ag(state, uav_index)
            if len(state.avail_uav_actions) != len(state.behavior_change):
                raise ValueError("The length of avail_uav_actions & behavior set do not match in get_uav_belief() - 1")
        elif state.use_DAP:
            state.uav_actions = ae.get_closest_actions(state, uav_index)
            JSAPState.total_sampling_time += time.perf_counter() - time1
        elif state.use_Learning:
            state.uav_actions = ae.get_uav_action_gnn(state, uav_index)
            JSAPState.total_sampling_time += time.perf_counter() - time1
        state.state_actions = [action for action in state.uav_actions]
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.TRAV:  # only at goal
        state.depth += 1
        if vertex.id == state.goalIDs[0]:
            state.uav_actions = [core.Action(target=state.goalIDs[0], rtype=param.RobotType.Drone)]
            state.uav_actions[0].update_pose((state.uavs[uav_index].cur_pose[0], state.uavs[uav_index].cur_pose[1]))
            state.uav_actions[0].update_robotID(uav_index)
        else:
            time1 = time.perf_counter()
            if state.use_AVP:
                # state.update_action_bc() # move the action bc update here from ground
                state.update_action_bc_networkX()
                state.uav_actions = ae.get_uav_action_2ag(state, uav_index)
                if len(state.avail_uav_actions) != len(state.behavior_change):
                    raise ValueError("The length of avail_uav_actions & behavior set do not match in get_uav_belief() - 2")
            elif state.use_DAP:
                state.uav_actions = ae.get_closest_actions(state, uav_index)
                JSAPState.total_sampling_time += time.perf_counter() - time1
            elif state.use_Learning:
                state.uav_actions = ae.get_uav_action_gnn(state, uav_index)
                JSAPState.total_sampling_time += time.perf_counter() - time1
            if len(state.uav_actions) == 0:
                state.uav_actions = [core.Action(target=state.goalIDs[0], rtype=param.RobotType.Drone)]
                state.uav_actions[0].update_pose((state.uavs[uav_index].cur_pose[0], state.uavs[uav_index].cur_pose[1]))
                state.uav_actions[0].update_robotID(uav_index)
        state.state_actions = [action for action in state.uav_actions]
        state.update_heuristic()
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.CHANCE:
        new_state_trav = get_new_uav_node(state, uav_index, blocked=False) # TRAVERSABLE
        new_state_block = get_new_uav_node(state, uav_index, blocked=True) # BLOCKED
        
        return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                    new_state_block: (vertex.block_prob, new_state_block.action_cost)}

def get_new_uav_node(state, uav_index, blocked=False):
    new_state = state.copy()
    new_state.depth += 1
    new_state.cur_ugv_idx = -1
    new_state.action_cost = state.action_cost
    edge = state.graph.get_poi(state.uavs[uav_index].last_node).neighbors
    edge = [edge[0]-1, edge[1]-1]
    if blocked:
        new_state.history.add_history(state.uavs[uav_index].action, param.EventOutcome.BLOCK)
        prob = 1.0
    else:
        new_state.history.add_history(state.uavs[uav_index].action, param.EventOutcome.TRAV)
        prob = 0.0
    new_state.pg_probabilities = ug.set_edge_probabilities(probs=np.array([prob]), edges=[edge], probabilities=state.pg_probabilities)
    for i, robot in enumerate(new_state.ugvs):
        if not robot.at_node: # reset if it is in middle of action
            robot.need_action = True # you don't want it to go to get_new_nodes_grobot (no node reached)
            robot.remaining_time = 0.0
            actions = get_ugv_action(new_state, i)
            if actions != []:
                new_state.ugvs_actions[i] = actions
            else:
                new_state.noway2goal = True
                new_state.action_cost = param.STUCK_COST
                new_state.ugvs_actions[i] = []
    new_state.update_heuristic()
    assert new_state.use_AVP == state.use_AVP
    assert new_state.max_uanum == state.max_uanum
    time1 = time.perf_counter()
    if new_state.use_AVP:
        new_state.update_action_bc_networkX()
        new_state.uav_actions = ae.get_uav_action_2ag(new_state, uav_index)
        if len(new_state.avail_uav_actions) != len(new_state.behavior_change):
            raise ValueError("The length of avail_uav_actions & behavior set do not match in get_uav_belief() - 3")
    elif new_state.use_DAP:        
        new_state.uav_actions = ae.get_closest_actions(new_state, uav_index)
        JSAPState.total_sampling_time += time.perf_counter() - time1
    elif new_state.use_Learning:
        new_state.uav_actions = ae.get_uav_action_gnn(new_state, uav_index)
        JSAPState.total_sampling_time += time.perf_counter() - time1
    if len(new_state.uav_actions)  == 0:
        state.uav_actions = [core.Action(target=state.goalIDs[0], rtype=param.RobotType.Drone)]
        state.uav_actions[0].update_pose((state.uavs[uav_index].cur_pose[0], state.uavs[uav_index].cur_pose[1]))
        state.uav_actions[0].update_robotID(uav_index)
    new_state.state_actions = [action for action in new_state.uav_actions]
    return new_state


def reset_uavs_action(state, robot_idx):
    for i, uav in enumerate(state.uavs):
        if not uav.need_action and uav.action.target == state.ugvs[robot_idx].action.target \
                    and uav.action.target != state.goalIDs[0]:
            uav.need_action = True 
            uav.remaining_time = 0.0

def _get_robot_that_finishes_first(state):
    time_remaining_uavs = []
    ugv_finish_first = True
    if len(state.uavs) > 0:
        for uav in state.uavs:
            if uav.last_node == state.goalIDs[0] and uav.remaining_time <= param.APPROX_TIME:
                continue
            time_remaining_uavs.append(uav.remaining_time)
    ugvs_remaining_times = []
    for i, ugv in enumerate(state.ugvs):
        if ugv.last_node == state.goalIDs[i] and ugv.remaining_time <= param.APPROX_TIME:
            continue
        ugvs_remaining_times.append(ugv.remaining_time)
    assert len(ugvs_remaining_times) > 0
    min_ugv_time = min(ugvs_remaining_times)
    if len(time_remaining_uavs)==0 or (min_ugv_time < min(time_remaining_uavs)-param.APPROX_TIME):
        remaining_times = [ugv.remaining_time for ugv in state.ugvs]
        idx = -1
        for ii, time in enumerate(remaining_times):
            if state.ugvs[ii].last_node != state.goalIDs[ii] and time ==min_ugv_time:
                idx = ii
                break
        assert idx >= 0
        return ugv_finish_first, idx, min_ugv_time # remaining_times.index(min_ugv_time), min_ugv_time
    else:
        assert len(state.uavs) > 0
        min_uav_time = min(time_remaining_uavs)
        remaining_times = [uav.remaining_time for uav in state.uavs]
        idx = -1
        for ii, time in enumerate(remaining_times):
            if time ==min_uav_time:
                idx = ii
                break
        assert idx >= 0
        ugv_finish_first = False
        return ugv_finish_first, idx, min_uav_time #remaining_times.index(min_uav_time), min_uav_time


def decsctp_rollout(state):
    if state.is_goal_state and not state.is_block_state:
        return 0.0
    if state.heuristic >= 0.0:
        return state.heuristic
    return state.update_heuristic()

# def update_prob_matrix(graph, matrix: np.ndarray, poi_id: int, block_prob: float):
#     edge = graph.get_poi(poi_id).neighbors
#     assert edge is not None
#     matrix[edge[0], edge[1]] = block_prob
#     matrix[edge[1], edge[0]] = block_prob
#     return matrix
