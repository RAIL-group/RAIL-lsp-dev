from sctp import graph as g
from sctp.utils import paths, plotting
import numpy as np
import random
from sctp import param, core
import pytest

def get_uav_action_2ag(state, uav_index):
    actions = []
    for act in list(state.behavior_change.keys()):
        state.action_values[act] = get_action_value(state.behavior_change[act], act, 
                                                    state.uavs[uav_index].cur_pose, state.graph)
    state.action_values = dict(sorted(state.action_values.items(), key=lambda item: item[1], reverse=True))
    actions = list(state.action_values.keys())[:min(state.max_uanum, len(state.action_values))]
    for action in actions:
        assert action in state.behavior_change
        assert action in state.action_values
        state.action_values.pop(action)
        state.behavior_change.pop(action)
        action.update_pose((state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))
        action.update_robotID(uav_index)    
    return actions



# def using_uav_action_values(state, uav_index):
#     if state.use_2AG: # adding information gain
#         if len(state.uav_actions) == 0 and len(state.uav_action_values) == 0:
#             state.uav_actions = [core.Action(target=state.goalID, rtype=param.RobotType.Drone, 
#                                         start_pose = (state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))]
#             state.state_actions = [action for action in state.uav_actions]
#         elif len(state.uav_actions) > 0:
#             for action in state.uav_actions:
#                 action.update_pose((state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))
#             state.state_actions = [action for action in state.uav_actions if action.target not in state.assigned_pois]
#     else:
#         if len(state.uav_actions) == 0:
#             state.uav_actions = [core.Action(target=state.goalID, rtype=param.RobotType.Drone, 
#                                         start_pose = (state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))]
#             state.state_actions = [action for action in state.uav_actions]
#         else:
#             for action in state.uav_actions:
#                 action.update_pose((state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))
#             state.state_actions = [action for action in state.uav_actions if action.target not in state.assigned_pois]


def get_single_behavior_change(graph, action, robot_edge, d0, d1, goalID, atNode, cur_heuristic, n_samples=100):
    # value if the action is passable
    block_value = 0.0
    pass_value = 0.0
    num_pois = len(graph.pois)
    num_vertices = len(graph.vertices)
    num_edges = len(graph.edges)
    for _ in range(n_samples):
        assert num_pois == len(graph.pois)
        assert num_vertices == len(graph.vertices)
        assert num_edges == len(graph.edges)
        pass_value += sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=False)
        block_value += sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=True)
    pass_value /= n_samples
    block_value /= n_samples
    aver_block = graph.get_poi(action.target).block_prob * block_value
    aver_pass = (1-graph.get_poi(action.target).block_prob) * pass_value
    return cur_heuristic - (aver_block + aver_pass)

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
    
def get_action_value(bc, action, drone_pose, graph):
    return bc - np.linalg.norm(np.array(drone_pose)-np.array(graph.get_poi(action.target).coord))/param.VEL_RATIO

def sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=False):
    block_pois = [poi.id for poi in graph.pois if poi.id != action.target and random.random() <= poi.block_prob ] 
    if block_edge:
        modified_graph = g.modify_graph(graph=graph, robot_edge=robot_edge, poiIDs=block_pois+[action.target])
    else:
        modified_graph = g.modify_graph(graph=graph, robot_edge=robot_edge, poiIDs=block_pois)
    if atNode:
        cost, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[0], goal=goalID)
        return cost if cost >= 0.0 else param.NOWAY_PEN
    else:
        cost0, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[0], goal=goalID)
        cost1, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[1], goal=goalID)
        assert (cost1 < 0) == (cost0 < 0)
        return min(cost0+d0, cost1+d1) if cost0 >= 0 else param.NOWAY_PEN


def _is_robot_goal_connected(graph, history, redge, goalID):
    block_pois = []
    for key, value in history.get_data().items():
        if value == param.EventOutcome.BLOCK:
            block_pois.append(key.target)
    new_graph = g.modify_graph(graph=graph, robot_edge=redge, poiIDs=block_pois)
    reach = paths.is_reachable(graph=new_graph, start=redge[0], goal=goalID)
    return reach
