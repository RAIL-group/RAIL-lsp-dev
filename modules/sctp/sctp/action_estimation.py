from sctp import graph as g
from sctp.utils import paths, plotting
import numpy as np
import random
import torch
from sctp import param, core
import heapq
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
            target_node = state.graph.get_poi(action.target)
            # target_node = [node for node in state.graph.pois if node.id == action.target][0]
            distance = 0.0
            for ugv in state.ugvs:
                ugv_pose = (ugv.cur_pose[0], ugv.cur_pose[1])
                distance += np.linalg.norm(np.array(ugv_pose) - np.array(target_node.coord))
            # distance = np.linalg.norm(np.array(uav_pose) - np.array(target_node.coord))
            action_dict.update({action: distance})
        # sorted_dict = dict(sorted(action_dict.items(), key=lambda item: item[1]))
        # actions = list(sorted_dict.keys())[:state.max_uanum]
        k = min(state.max_uanum, len(action_dict))
        actions = [
            action for action, _ in
            # heapq.nsmallest(k, state.action_values.items(), key=lambda x: x[1])
            heapq.nsmallest(k, action_dict.items(), key=lambda x: x[1])
        ]

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
    if uav_index is None:
        raise ValueError("UAV index is None - get_uav_action_gnn")

    state.action_values.clear()
    edge_dict = {}

    # ── GNN per UGV with individual caching ──
    for i, ugv in enumerate(state.ugvs):
        if ugv.at_node and ugv.last_node == state.goalIDs[i]:
            continue

        goal  = state.goalIDs[i] - 1
        start = _get_ugv_start(ugv, state.graph)

        # Cache key is per-UGV — only what affects THIS UGV's GNN call
        cache_key = (
            i,                                        # UGV index
            start,                                    # UGV start node
            goal,                                     # UGV goal node
            tuple(state.pg_probabilities.flatten()),  # shared graph belief
        )

        if state.gnn_cache is not None and cache_key in state.gnn_cache:
            ugv_edge_dict = state.gnn_cache[cache_key]   # ← cache hit
        else:
            # ── Build data and run GNN ──
            prob_graph = ug.ProbabilisticGraph(
                positions=state.pg_positions,
                adjacency=state.pg_adjacency,
                probabilities=state.pg_probabilities
            )
            edges = [[edge[0], edge[1]] for edge in state.edges]
            data  = create_graph_datum(
                graph=prob_graph, edges=edges,
                start=start, goal=goal,
                values=np.zeros(len(state.graph.pois))
            )
            data = graphdata_to_pyg(data, state.device)

            with torch.no_grad():
                pred, _ = state.model(
                    x          = data.x,
                    edge_index = data.edge_index,
                    edge_attr  = data.edge_attr,
                )

            pred_cpu = pred.cpu()
            src = data.edge_index[0].cpu()
            dst = data.edge_index[1].cpu()
            probs = data.edge_attr[:,1].cpu()

            ugv_edge_dict = {}
            for j in range(pred_cpu.shape[0]):
                u, v = src[j].item(), dst[j].item()
                if v < u:
                    u, v = v, u
                # ugv_edge_dict[(u, v)] = ugv_edge_dict.get((u, v), 0.0) + pred_cpu[j].item()
                block_prob = probs[j].item()
                ugv_edge_dict[(u, v)] =  block_prob*(1.0-block_prob)*pred_cpu[j].item()

            # ── Store per-UGV result ──
            if state.gnn_cache is not None:
                state.gnn_cache[cache_key] = ugv_edge_dict   # ← cache miss, store

        # Accumulate across UGVs
        for edge, val in ugv_edge_dict.items():
            edge_dict[edge] = edge_dict.get(edge, 0.0) + val

    # ── Score and rank UAV actions ──
    drone_pose = np.array(state.uavs[uav_index].cur_pose)
    for action in state.avail_uav_actions:
        target_node = state.graph.get_poi(action.target)
        edge = tuple(sorted((target_node.neighbors[0] - 1, target_node.neighbors[1] - 1)))

        assert edge in edge_dict, (
            f"Edge {edge} not found in edge_dict. Available: {list(edge_dict.keys())}"
        )
        dist = np.linalg.norm(drone_pose - np.array(target_node.coord))
        state.action_values[action] = edge_dict[edge] * param.VEL_RATIO / dist
        # print(f"Action {action} | Edge {edge} | GNN Value {edge_dict[edge]:.4f} | Dist {dist:.4f} | Final Value {state.action_values[action]:.4f}")
    # ── Get top K without full sort ──
    k = min(state.max_uanum, len(state.action_values))
    actions = [
        action for action, _ in
        heapq.nlargest(k, state.action_values.items(), key=lambda x: x[1])
    ]

    for action in actions:
        action.update_pose((drone_pose[0], drone_pose[1]))
        action.update_robotID(uav_index)
    # if state.depth>20:
    #     print(f"The depth is {state.depth} with total of {len(state.avail_uav_actions)} available UAV actions.")
    #     print(f"Selected UAV Actions: {[act.target for act in actions]}")
    #     print("-------------------------------------------------------")
    return actions

def get_bestAction_gnn(edges, graph, startID, goalID, \
            device, gnn_model, drone_pose, actions): 
    pg_positions = ug.get_vertex_positions(graph.vertices)
    pg_adjacency, pg_probabilities = ug.create_adj_prob_matrices(edges, pg_positions)
    goal  = goalID - 1
    start = startID - 1

    # ── Build data and run GNN ──
    prob_graph = ug.ProbabilisticGraph(
        positions=pg_positions,
        adjacency=pg_adjacency,
        probabilities=pg_probabilities
    )
    es = [[edge[0], edge[1]] for edge in edges]
    data  = create_graph_datum(
        graph=prob_graph, edges=es,
        start=start, goal=goal,
        values=np.zeros(len(es))
    )
    data = graphdata_to_pyg(data, device)

    with torch.no_grad():
        pred, _ = gnn_model(
            x          = data.x,
            edge_index = data.edge_index,
            edge_attr  = data.edge_attr,
        )
    pred_cpu = pred.cpu()
    src = data.edge_index[0].cpu()
    dst = data.edge_index[1].cpu()
    prob = data.edge_attr[1].cpu()
    ugv_edge_dict = {}
    for j in range(pred_cpu.shape[0]):
        u, v = src[j].item(), dst[j].item()
        if v < u:
            u, v = v, u
        ugv_edge_dict[(u, v)] = ugv_edge_dict.get((u, v), 0.0) + pred_cpu[j].item()
        # block_prob = prob[j].item()
        # ugv_edge_dict[(u, v)] = block_prob*(1-block_prob)*pred_cpu[j].item()


    action_values = {}
    
    for action in actions:
        target_node = graph.get_poi(action.target)
        if (target_node.block_prob ==0.0 or target_node.block_prob == 1.0):
            continue
        edge = tuple(sorted((target_node.neighbors[0] - 1, target_node.neighbors[1] - 1)))

        dist = np.linalg.norm(drone_pose - np.array(target_node.coord))
        action_values[action] = ugv_edge_dict[edge] * param.VEL_RATIO / dist
    actions = [
        action for action, _ in
        heapq.nlargest(1, action_values.items(), key=lambda x: x[1])
    ]
    action = actions[0]
    action.update_pose((drone_pose[0], drone_pose[1]))

    return action

def _get_ugv_start(ugv, graph):
    if ugv.at_node:
        if ugv.last_node in graph.poiIDs:
            start = ugv.pl_vertex-1
        else:
            start = ugv.last_node-1
    else:
        start = ugv.edge[0]-1 if ugv.edge[0] not in graph.poiIDs else ugv.edge[1]-1
    return start

def get_ugvs_heuristic(state):
    heuristic = 0.0
    pg = ug.ProbabilisticGraph(positions=state.pg_positions, adjacency=state.pg_adjacency, \
                                        probabilities=state.pg_probabilities)
    for i, ugv in enumerate(state.ugvs):
        if ugv.at_node and ugv.last_node == state.goalIDs[i]:
            continue
        if ugv.at_node:
            if ugv.last_node in state.graph.poiIDs: # at a poi
                poi = state.graph.get_poi(ugv.last_node)
                assert len(poi.neighbors) == 2, f"POI {poi.id} has {len(poi.neighbors)} neighbors"
                start1 = poi.neighbors[0]
                start2 = poi.neighbors[1]
                if pg.probabilities[start1-1, start2-1] == 1.0:
                    bc = _cal_optimistic_path(ugraph=pg, start=ugv.pl_vertex-1, goalID=state.goalIDs[i]-1)
                else:
                    bc1 = _cal_optimistic_path(ugraph=pg, start=start1-1, goalID=state.goalIDs[i]-1)
                    bc2 = _cal_optimistic_path(ugraph=pg, start=start2-1, goalID=state.goalIDs[i]-1)
                    dist = pg.adjacency[start1-1, start2-1]/2.0
                    bc = min(bc1 + dist, bc2 + dist)
            else: # a node
                start = ugv.last_node
                bc = _cal_optimistic_path(ugraph=pg, start=start-1, goalID=state.goalIDs[i]-1)
        else:
            edge = ugv.edge
            # start = ugv.pl_vertex if ugv.last_node in state.graph.poiIDs else ugv.last_node
            if edge[0] in state.graph.poiIDs:
                start1 = edge[1] # ugv.pl_vertex-1
                poiId = edge[0]
            else:
                start1 = edge[0]
                poiId = edge[1]
            poi = state.graph.get_poi(poiId)
            start2 = poi.neighbors[0] if poi.neighbors[0] != start1 else poi.neighbors[1]
            if pg.probabilities[start1-1, start2-1] == 1.0:
                bc = _cal_optimistic_path(ugraph=pg, start=start1-1, goalID=state.goalIDs[i]-1) 
            else:
                bc1 = _cal_optimistic_path(ugraph=pg, start=start1-1, goalID=state.goalIDs[i]-1) 
                bc2 = _cal_optimistic_path(ugraph=pg, start=start2-1, goalID=state.goalIDs[i]-1) 
                dist1 = np.linalg.norm(np.array(state.vertices_map[start1].coord) - np.array(ugv.cur_pose))
                dist2 = np.linalg.norm(np.array(state.vertices_map[start2].coord) - np.array(ugv.cur_pose))
                bc = min(bc1 + dist1, bc2 + dist2)
        heuristic += bc
    return heuristic

def _cal_optimistic_path(ugraph, start, goalID):
    # the start, goalID, edge are 0-indexed for the underlying graph
    adjacency_matrix = ug.get_optimistic_adj_matrix(ugraph.probabilities, ugraph.adjacency)
    val = ug.compute_shortest_path_length(adjacency_matrix, start=start, end=goalID)
    return val if val >=0 else NOWAY_PEN