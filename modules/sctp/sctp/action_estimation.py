from sctp import graph as g
from sctp.utils import paths, plotting
import numpy as np
import random
import torch
from sctp import param, core
from sctp.utils import underlying_graph as ug
from sctp.scripts.data_gen import GraphData, create_graph_datum, graphdata_to_pyg


NOWAY_PEN = 500.0

def get_closest_actions(state, uav_idx):
    action_dict = {}
    actions = []
    assert state.use_AVP == False
    uav = state.uavs[uav_idx]
    assert state.avail_uav_actions is not None
    if len(state.avail_uav_actions) <= state.max_uanum:
        actions = state.avail_uav_actions
        # return state.avail_uav_actions
    else:
        for action in state.avail_uav_actions:
            target_node = [node for node in state.graph.pois if node.id == action.target][0]
            distance = 0.0
            for ugv in state.ugvs:
                ugv_pose = (ugv.cur_pose[0], ugv.cur_pose[1])
                distance += np.linalg.norm(np.array(ugv_pose) - np.array(target_node.coord))
            # distance = np.linalg.norm(np.array(uav_pose) - np.array(target_node.coord))
            action_dict.update({action: distance})
        sorted_dict = dict(sorted(action_dict.items(), key=lambda item: item[1]))
        actions = list(sorted_dict.keys())[:state.max_uanum]
    for action in actions:
        action.update_pose((uav.cur_pose[0],uav.cur_pose[1]))
        action.update_robotID(uav_idx)
    return actions

def get_uav_action_2ag(state, uav_index):
    actions = []
    state.action_values.clear()
    if len(state.behavior_change) != len(state.avail_uav_actions):
        raise ValueError("Behavior change and available uav actions size mismatch - get_uav_action_2ag")
    for action, value in state.behavior_change.items():
        state.action_values[action] = get_action_value(bc=value, action=action, \
                        drone_pose=state.uavs[uav_index].cur_pose, graph=state.graph)    
    state.action_values = dict(sorted(state.action_values.items(), key=lambda item: item[1], reverse=True))
    actions = list(state.action_values.keys())[:min(state.max_uanum, len(state.action_values))]
    for action in actions:
        assert action in state.behavior_change
        assert action in state.action_values
        action.update_pose((state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))
        action.update_robotID(uav_index) 
    return actions

def get_ugvs_behavior_change(state, action):
    act_value = 0.0
    for i, ugv in enumerate(state.ugvs):
        if ugv.at_node and ugv.last_node == state.goalIDs[i]:
            continue
        redge = [ugv.last_node, ugv.pl_vertex]
        d1 = np.linalg.norm(np.array(ugv.cur_pose)-np.array(state.vertices_map[redge[0]].coord))
        d2 = np.linalg.norm(np.array(ugv.cur_pose)-np.array(state.vertices_map[redge[1]].coord))
        bc = get_single_behavior_change(graph=state.graph, action=action, robot_edge=redge,
                                    d0=d1, d1=d2, goalID=state.goalIDs[i], atNode=ugv.at_node,
                                    cur_heuristic=state.heuristic, n_samples=state.sampling_maps)
        act_value += bc
    return act_value

def get_single_behavior_change(graph, action, robot_edge, d0, d1, goalID, atNode, cur_heuristic, n_samples=60):
    # value if the action is passable
    block_value = 0.0
    pass_value = 0.0
    for _ in range(n_samples):
        pass_value += sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=False)
        block_value += sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=True)
    pass_value /= n_samples
    block_value /= n_samples
    aver_block = graph.get_poi(action.target).block_prob * block_value
    aver_pass = (1-graph.get_poi(action.target).block_prob) * pass_value
    return (block_value - pass_value)

def sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=False):
    block_pois = [poi.id for poi in graph.pois if poi.id != action.target and random.random() <= poi.block_prob ] 
    if block_edge:
        modified_graph = g.modify_graph(graph=graph, robot_edge=robot_edge, poiIDs=block_pois+[action.target])
    else:
        modified_graph = g.modify_graph(graph=graph, robot_edge=robot_edge, poiIDs=block_pois)
    
    if atNode:
        cost, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[0], goal=goalID)
        return cost if cost >= 0.0 else NOWAY_PEN
    else:
        cost0, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[0], goal=goalID)
        cost1, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[1], goal=goalID)
        assert (cost1 < 0) == (cost0 < 0)
        return min(cost0+d0, cost1+d1) if cost0 >= 0 else NOWAY_PEN

    
def get_action_value(bc, action, drone_pose, graph):
    # return bc - np.linalg.norm(np.array(drone_pose)-np.array(graph.get_poi(action.target).coord))/param.VEL_RATIO
    poi = graph.get_poi(action.target)
    return bc*poi.block_prob*(1.0-poi.block_prob)*param.VEL_RATIO/np.linalg.norm(np.array(drone_pose)-np.array(poi.coord))

def get_ugvs_bc_networkX(state, action, pg):
    act_value = 0.0
    neighbors = state.graph.get_poi(action.target).neighbors
    action_edge = [[neighbors[0]-1, neighbors[1]-1]]
    for i, ugv in enumerate(state.ugvs):
        if ugv.at_node and ugv.last_node == state.goalIDs[i]:
            continue
        if ugv.at_node:
            if ugv.last_node in state.graph.poiIDs: # at a poi
                poi = state.graph.get_poi(ugv.last_node)
                start1 = poi.neighbors[0]-1
                start2 = poi.neighbors[1]-1
                if pg.probabilities[start1, start2] == 1.0:
                    bc = get_single_bc_networkX(ugraph=pg, action_edge=action_edge, start=ugv.pl_vertex-1,
                                    goalID=state.goalIDs[i]-1, n_samples=state.sampling_maps)
                else:
                    bc1 = get_single_bc_networkX(ugraph=pg, action_edge=action_edge, start=start1,
                                    goalID=state.goalIDs[i]-1, n_samples=state.sampling_maps)
                    bc2 = get_single_bc_networkX(ugraph=pg, action_edge=action_edge, start=start2,
                                    goalID=state.goalIDs[i]-1, n_samples=state.sampling_maps)
                    dist = pg.adjacency[start1, start2]/2.0
                    bc = min(bc1 + dist, bc2 + dist)
            else: # a node
                start = ugv.last_node
                bc = get_single_bc_networkX(ugraph=pg, action_edge=action_edge, start=start-1,
                                    goalID=state.goalIDs[i]-1, n_samples=state.sampling_maps)
        else:
            start = ugv.pl_vertex if ugv.last_node in state.graph.poiIDs else ugv.last_node
            bc = get_single_bc_networkX(ugraph=pg, action_edge=action_edge, start=start-1,
                                    goalID=state.goalIDs[i]-1, n_samples=state.sampling_maps)
        act_value += bc
    return act_value

def get_single_bc_networkX(ugraph, action_edge, start, goalID, n_samples=60):
    # the start, goalID, edge are 0-indexed for the underlying graph
    block_value = 0.0
    pass_value = 0.0
    pass_probs = ug.set_edge_probabilities(probs=np.array([0.0]), edges=action_edge, probabilities=ugraph.probabilities)
    block_probs = ug.set_edge_probabilities(probs=np.array([1.0]), edges=action_edge, probabilities=ugraph.probabilities)
    for _ in range(n_samples):        
        pass_sample = ug.sample_graph(prob_graph=ugraph, probs=pass_probs)
        val = ug.compute_shortest_path_length(pass_sample, start=start, end=goalID)
        pass_value += val if val >=0 else NOWAY_PEN
        block_sample = ug.sample_graph(prob_graph=ugraph, probs=block_probs)
        val = ug.compute_shortest_path_length(block_sample, start=start, end=goalID)
        block_value += val if val >=0 else NOWAY_PEN
    pass_value /= n_samples
    block_value /= n_samples
    return (block_value - pass_value)

def get_uav_action_gnn(state, uav_index):
    actions = []
    state.action_values.clear()
    
    prob_graph = ug.ProbabilisticGraph(
            positions=state.pg_positions,
            adjacency=state.pg_adjacency,
            probabilities=state.pg_probabilities
        )
    
    for i, ugv in enumerate(state.ugvs):
        if ugv.at_node and ugv.last_node == state.goalIDs[i]:
            continue
        goal = state.goalIDs[i]-1
        if ugv.at_node:
            if ugv.last_node in state.graph.poiIDs:
                start = ugv.pl_vertex-1
            else:
                start = ugv.last_node-1
        else:
            start = ugv.edge[0]-1 if ugv.edge[0] not in state.graph.poiIDs else ugv.edge[1]-1
            
        edges = [[edge[0], edge[1]] for edge in state.edges]
        data =  create_graph_datum(graph=prob_graph, edges=edges, start=start, goal=goal, values=np.array([0.0]*len(state.graph.pois)))  
        
        data = graphdata_to_pyg(data, state.device)
        with torch.no_grad():
            pred, _ = state.model(
                x          = data.x,
                edge_index = data.edge_index,
                edge_attr  = data.edge_attr,
            )   # [E]

        pred_cpu   = pred.cpu()
        E          = pred_cpu.shape[0]

        src = data.edge_index[0].cpu()   # [E]
        dst = data.edge_index[1].cpu()   # [E]
        edge_dict = {}
        for i in range(E):
            u      = src[i].item()
            v      = dst[i].item()
            pr     = pred_cpu[i].item()
            if v < u:
                u, v = v, u
            if (u,v) in edge_dict:
                edge_dict[(u,v)] = +pr
            else:
                edge_dict[(u,v)] = pr
            
    drone_pose = state.uavs[uav_index].cur_pose
    for action in state.avail_uav_actions:
        target_node = [node for node in state.graph.pois if node.id == action.target][0]
        edge = tuple(sorted(target_node.neighbors))
        edge = (edge[0]-1, edge[1]-1) if edge[0] < edge[1] else (edge[1]-1, edge[0]-1)
        assert edge in edge_dict, f"Edge {edge} not found in edge_dict. Available edges: {list(edge_dict.keys())}"
        act_value = edge_dict[edge]        
        state.action_values[action] = act_value*param.VEL_RATIO/np.linalg.norm(np.array(drone_pose)-np.array(target_node.coord))
    
    state.action_values = dict(sorted(state.action_values.items(), key=lambda item: item[1], reverse=True))
    actions = list(state.action_values.keys())[:min(state.max_uanum, len(state.action_values))]
    for action in actions:
        assert action in state.action_values
        action.update_pose((state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))
        action.update_robotID(uav_index) 
    return actions
    