from sctp import graph as g
from sctp.utils import paths, plotting
import numpy as np
import random
from sctp import param, core
from sctp import action_estimation as ae
from sctp.utils import underlying_graph as ug
from typing import Dict, List
import time

class MAction(object):
    def __init__(self, start: int, sub_targets: List[int], distances: List[float], \
                    robotID: int =0, rtype=param.RobotType.Ground):
        self.start = start
        self.sub_targets = sub_targets
        self.target = self.sub_targets[-1]
        self.distances = distances
        self.total_dist = sum(self.distances) if len(self.distances) >0 else 0.0
        self.robotID = robotID
        self.rtype = rtype
    def update_robotID(self, robotID: int):
        self.robotID = robotID
    def copy(self):
        return MAction(start=self.start, sub_targets=self.sub_targets.copy(), \
                        distances=self.distances.copy(), robotID=self.robotID, rtype=self.rtype) 
    def __eq__(self, other):
        return self.target == other.target 
    def __hash__(self):
        return hash(self.target)
    def __str__(self):
        return f"MAction from {self.start} to {self.target} via {self.sub_targets} costs of {self.total_dist:.2f} meters"


class MCState(object):
    # total_sampling_time = 0.0
    # @classmethod
    # def reset_sampling_time(cls):
    #     cls.total_sampling_time = 0.0
    def __init__(self, graph=None, goalIDs=[], ugvs=[], iscopy=False):        
        self.action_cost = 0.0
        self.heuristic = -1.0
        self.depth = 0
        self.state_actions = []
        self.use_OptiHeur = True
        self.noway2goal = False
        self.cur_ugv_idx = -1
        self.action_values = dict() # map action to its value
        
        if not iscopy: # the first state
            # need to filter the visited POIs
            assert goalIDs != [] and graph is not None and len(ugvs) == 1
            self.graph = graph
            self.goalIDs = goalIDs
            self.history = core.History()
            self.vertices_map = {v.id: v for v in self.graph.vertices + self.graph.pois}
            self.neighbors_map = {v.id: v.neighbors for v in self.graph.vertices + self.graph.pois}
            self.edge_poi_map = {tuple(sorted(poi.neighbors)): poi.id for poi in self.graph.pois}
            self.init_history()
            self.ugvs = ugvs
            # set up underlying graph for sampling
            edges = ug.get_initial_edges(self.graph)
            self.pg_positions = ug.get_vertex_positions(self.graph.vertices)
            self.pg_adjacency, self.pg_probabilities = ug.create_adj_prob_matrices(edges, self.pg_positions)
            self.ugvs_actions = [[] for _ in range(len(self.ugvs))]
            for i, ugv in enumerate(self.ugvs):
                ugv.need_action = True
                if ugv.last_node == self.goalIDs[i]:
                    ugv_actions = [MAction(start=ugv.last_node, sub_targets= [ugv.last_node], times=[0.0], robotID=i)]
                else:
                    ugv_actions = get_macro_actions(self, i)
                    if ugv_actions == []:
                        self.noway2goal = True
                        self.action_cost = param.STUCK_COST
                # ugv.visited_vertices.append(ugv.last_node)                
                self.ugvs_actions[i] = ugv_actions
            self.cur_ugv_idx = 0
            idx = [i for i, robot in enumerate(self.ugvs) if robot.need_action==True]
            assert len(idx) > 0
            self.state_actions = [action for action in self.ugvs_actions[idx[0]]] # what if this ugv has reached goal?
            self.update_heuristic()
            
    def get_actions(self):
        return self.state_actions
    
    def init_history(self):
        for vertex in self.graph.vertices+self.graph.pois:
            action = MAction(start=vertex.id, sub_targets=[vertex.id], distances=[0.0])
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
    
        
    @property
    def is_goal_state(self):
        return all ([ugv.last_node == self.goalIDs[i] for i, ugv in enumerate(self.ugvs)]) or self.noway2goal

    @property
    def is_block_state(self):
        return self.noway2goal

    def copy(self):
        new_state = MCState(iscopy=True)
        new_state.cur_ugv_idx = self.cur_ugv_idx
        new_state.heuristic = self.heuristic
        new_state.vertices_map = self.vertices_map.copy()
        new_state.neighbors_map = self.neighbors_map.copy()
        new_state.edge_poi_map = self.edge_poi_map.copy()
        new_state.depth = self.depth
        new_state.graph = self.graph
        new_state.goalIDs = self.goalIDs.copy()
        new_state.use_OptiHeur = self.use_OptiHeur
        
        new_state.action_cost = 0.0
        new_state.history = self.history.copy()
        # save the underlying graph
        new_state.pg_positions = self.pg_positions
        new_state.pg_adjacency = self.pg_adjacency
        new_state.pg_probabilities = self.pg_probabilities
        new_state.state_actions = []
        # copy the robot
        new_state.ugvs = [ugv.copy() for ugv in self.ugvs]
        
        new_state.ugvs_actions = [[action.copy() for action in ugv_actions] for ugv_actions in self.ugvs_actions]
            
        return new_state
        
    def transition(self, action):
        temp_state = self.copy()
        assert temp_state.cur_ugv_idx > -1
        ugv_needs_action = [ugv.need_action for ugv in temp_state.ugvs]
        assert any(ugv_needs_action) == True
        ugv_idx = action.robotID
        temp_state.ugvs[ugv_idx].retarget(action)
    
        return advance_state(temp_state)


def create_marco_action(state, path, target, ugv_idx) -> MAction:
    start = path[0]
    distances = []
    sub_targets = []
    # print(f"The path is {path}")
    for i in range(1, len(path)):
        # print (f"Path step from {path[i-1]} to {path[i]}")
        dist = np.linalg.norm(np.array(state.vertices_map[path[i-1]].coord) - np.array(state.vertices_map[path[i]].coord))
        if path[i-1] in state.graph.poiIDs:
            sub_targets.append(path[i])
            distances.append(dist)
        else:
            poi = state.edge_poi_map.get(tuple(sorted([path[i-1], path[i]])), None)
            sub_targets.extend([poi, path[i]])    
            distances.extend([dist/2, dist/2])
    if target != path[-1]:
        dist = np.linalg.norm(np.array(state.vertices_map[path[-1]].coord) - np.array(state.vertices_map[target].coord))
        sub_targets.append(target)
        distances.append(dist)
    return MAction(start=start, sub_targets=sub_targets, distances=distances, robotID=ugv_idx)

def get_avail_mactions(state: MCState, uncertain_pois: List[int], ugv_idx: int) -> List[int]:
    mactions = []
    # update the underlying graph based on the history
    edges = []
    probs = []
    for key, value in state.history.get_data().items():
        if key.target not in state.graph.poiIDs:
            continue
        neighbors = state.neighbors_map[key.target]
        if value == param.EventOutcome.BLOCK:
            edges.append([neighbors[0]-1, neighbors[1]-1])
            probs.append(1.0)
        elif value == param.EventOutcome.TRAV:
            edges.append([neighbors[0]-1, neighbors[1]-1])
            probs.append(0.0)
    updated_probs = ug.set_edge_probabilities(probs=np.array(probs), edges=edges, probabilities=state.pg_probabilities)
    
    # get the certain graph based on the history
    certain_adjacency = ug.get_certain_adj_matrix(prob_matrix=updated_probs, adj_matrix=state.pg_adjacency)
    cur_node = state.ugvs[ugv_idx].last_node
    for poi in uncertain_pois: # all possible targets
        target_neighbors = state.neighbors_map[poi]
        path = get_best_path(state, cur_node, poi, ugv_idx, certain_adjacency, target_neighbors)
        if path != []:
            maction = create_marco_action(state, path=path, target=poi, ugv_idx=ugv_idx)
            mactions.append(maction)
    # macro-action reach goal directly
    target_neighbors = [state.goalIDs[ugv_idx]]
    path = get_best_path(state, cur_node, state.goalIDs[ugv_idx], ugv_idx, certain_adjacency, target_neighbors)
    if path != []:
        maction = create_marco_action(state, path=path, target=state.goalIDs[ugv_idx], ugv_idx=ugv_idx)
        mactions.append(maction)
    return mactions

def get_best_path(state, cur_node, poi, ugv_idx, certain_adjacency, target_neighbors):
    assert state.ugvs[ugv_idx].at_node == True
    if cur_node in state.graph.poiIDs:
        start_neighbors = state.neighbors_map[cur_node]
        # print(f"Current node {cur_node} is a POI, its neighbors are {start_neighbors}")
        path1, cost1 = ug.get_shortest_path_from_vertices(certain_adjacency, start=start_neighbors[0], targets=target_neighbors)
        path2, cost2 = ug.get_shortest_path_from_vertices(certain_adjacency, start=start_neighbors[1], targets=target_neighbors)
        if cost1 < 0 and cost2 < 0:
            path = []
        elif cost1 < 0:
            path = [cur_node] + path2
        elif cost2 < 0:
            path = [cur_node] + path1
        else:
            path = [cur_node] + path1 if cost1 <= cost2 else [cur_node] + path2
    else:
        path, _ = ug.get_shortest_path_from_vertices(certain_adjacency, start=cur_node, targets=target_neighbors)
    return path
        


def get_avail_pois(state: MCState) -> List[int]:
    avail_pois = []
    for poi in state.graph.pois:
        action = MAction(start=poi.id, sub_targets=[poi.id], distances=[0.0])
        if state.history.get_action_outcome(action) == param.EventOutcome.CHANCE:
            avail_pois.append(poi.id)
    return avail_pois
    
def get_macro_actions(state: MCState, ugv_idx: int) -> List[MAction]:
    # get all uncertain pois
    avail_pois = get_avail_pois(state)
    return get_avail_mactions(state, avail_pois, ugv_idx)



def advance_state(state):
    # 1. if any robot needs action, determine its actions then return
    robots_need_action = [robot.need_action for robot in state.ugvs]
    if any(robots_need_action):
        # set action for this state        
        robot_idx = robots_need_action.index(True)
        state.ugvs_actions[robot_idx] = get_macro_actions(state, robot_idx)
        if len(state.ugvs_actions[robot_idx]) == 0:
            state.noway2goal = True
            state.action_cost = param.STUCK_COST
            state.state_actions = []
        else:
            assert all ([action.robotID == robot_idx for action in state.ugvs_actions[robot_idx]])
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
    if robot_reach_first:
        return get_ugv_belief(state, last_nodes=last_nodes, robot_idx = rd_index, last_edges=edges)
    else:
        return get_uav_belief(state=state, uav_index=rd_index)
    

def get_ugv_belief(state, last_nodes, robot_idx, last_edges): # need to work on this.
    vertex_status = state.history.get_action_outcome(state.ugvs[robot_idx].action)
    vertex = [node for node in state.graph.vertices+state.graph.pois if node.id == state.ugvs[robot_idx].action.target][0]
    # state.ugvs[robot_idx].visited_vertices.append(state.ugvs[robot_idx].last_node)
    # state.v_vertices[state.ugvs[robot_idx].last_node] = state.v_vertices.get(state.ugvs[robot_idx].last_node, 0) + 1
    assert state.ugvs[robot_idx].at_node == True
    state.cur_ugv_idx = robot_idx
    # One UGV is processed one a time
    for i, ugv in enumerate(state.ugvs):
        if i != robot_idx:
            ugv.need_action = False
    if vertex_status == param.EventOutcome.BLOCK:
        assert 1== 0, "This should not never happen"
        state.depth += 1
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.TRAV: # should be the goal
        state.ugvs_actions[robot_idx] = [MAction(start=state.ugvs[robot_idx].last_node, distances=[0.0], \
                                        sub_targets=[state.ugvs[robot_idx].last_node], robotID=robot_idx)]
        state.state_actions = [action for action in state.ugvs_actions[robot_idx] ]
        state.update_heuristic()
        state.depth += 1
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.CHANCE:
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
        new_state.history.add_history(state.ugvs[robot_idx].action, param.EventOutcome.BLOCK)
    else:
        new_state.history.add_history(state.ugvs[robot_idx].action, param.EventOutcome.TRAV)
    new_state.ugvs_actions[robot_idx] = get_macro_actions(new_state, robot_idx)
    if len(new_state.ugvs_actions[robot_idx]) == 0:
        new_state.noway2goal = True
        new_state.action_cost = param.STUCK_COST
            
    for i, robot in enumerate(new_state.ugvs): # reset all other UGVs if they are in the middle of their action
        if i != robot_idx and not robot.at_node: 
            robot.need_action = True
            robot.remaining_time = 0.0
            new_state.ugvs_actions[i] =  get_macro_actions(new_state, i)
            if new_state.ugvs_actions[i] == []:
                new_state.noway2goal = True
                new_state.action_cost = param.STUCK_COST
    new_state.depth += 1
    new_state.update_heuristic()
    new_state.state_actions = [action for action in new_state.ugvs_actions[robot_idx]]    
    return new_state

def get_uav_belief(state, uav_index):
    assert 1==0
    return None


def _get_robot_that_finishes_first(state):
    # time_remaining_umax(len(self.travel_history)-2, 0)avs = []
    ugv_finish_first = True
    # if len(state.uavs) > 0:
    #     for uav in state.uavs:
    #         if uav.last_node == state.goalIDs[0] and uav.remaining_time <= param.APPROX_TIME:
    #             continue
    #         time_remaining_uavs.append(uav.remaining_time)
    ugvs_remaining_times = []
    for i, ugv in enumerate(state.ugvs):
        if ugv.last_node == state.goalIDs[i] and ugv.remaining_time <= param.APPROX_TIME:
            continue
        ugvs_remaining_times.append(ugv.remaining_time)
    assert len(ugvs_remaining_times) > 0
    min_ugv_time = min(ugvs_remaining_times)
    # if len(time_remaining_uavs)==0 or (min_ugv_time < min(time_remaining_uavs)-param.APPROX_TIME):
    remaining_times = [ugv.remaining_time for ugv in state.ugvs]
    idx = remaining_times.index(min_ugv_time)
    # for ii, time in enumerate(remaining_times):
    #     if state.ugvs[ii].last_node != state.goalIDs[ii] and time ==min_ugv_time:
    #         idx = ii
    #         break
    # assert idx >= 0
    return ugv_finish_first, idx, min_ugv_time
    # else:
    #     assert len(state.uavs) > 0
    #     min_uav_time = min(time_remaining_uavs)
    #     remaining_times = [uav.remaining_time for uav in state.uavs]
    #     idx = -1
    #     for ii, time in enumerate(remaining_times):
    #         if time ==min_uav_time:
    #             idx = ii
    #             break
    #     assert idx >= 0
    #     ugv_finish_first = False
    #     return ugv_finish_first, idx, min_uav_time #remaining_times.index(min_uav_time), min_uav_time


def decsctp_rollout(state):
    if state.is_goal_state and not state.is_block_state:
        return 0.0
    if state.heuristic >= 0.0:
        return state.heuristic
    return state.update_heuristic()