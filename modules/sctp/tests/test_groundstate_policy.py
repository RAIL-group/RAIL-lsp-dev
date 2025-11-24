import pytest
import argparse
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.robot import Robot
from pouct_planner import core as policy
from sctp import param, gstate_dec
# from sctp.utils import plotting, paths
import matplotlib.pyplot as plt

def test_groundstate_policy_lgraph():
    print()
    C=200.0
    num_iter=10
    n_maps=500    
    start, goal, l_graph = graphs.linear_graph_unc()
    robot = Robot(position=[0.0, 0.0], cur_node=start.id, at_node=True, robot_type=param.RobotType.Ground)
    robot.unfinished_action = None
    state = gstate_dec.GroundState(graph=l_graph, goalID=goal.id, robot=robot, n_maps=n_maps)

    best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iter, rollout_fn= core.sctp_rollout3)
    print(best_action)
    print(cost)
    print([[a.target, a.start_pose] for a in path_cost[0]])  
    print([c for c in path_cost[1]])  


def test_groundstate_policy_dgraph():
    print()
    C=200.0
    num_iter=2000
    sampling_maps=500    
    start, goal, graph = graphs.disjoint_unc()
    robot = Robot(position=[0.0, 0.0], cur_node=start.id, at_node=True, robot_type=param.RobotType.Ground)
    robot.unfinished_action = None
    state = gstate_dec.GroundState(graph=graph, goalID=goal.id, robot=robot, n_maps=sampling_maps)
    
    best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iter, rollout_fn= core.sctp_rollout3)
    print(best_action)
    print(cost)
    print([[a.target, a.start_pose] for a in path_cost[0]])  
    print([c for c in path_cost[1]])  

def test_groundstate_policy_sgraph():
    print()
    C=200.0
    num_iter=500
    sampling_maps=500    
    start, goal, graph = graphs.s_graph_unc()
    robot = Robot(position=[0.0, 0.0], cur_node=start.id, at_node=True, robot_type=param.RobotType.Ground)
    robot.unfinished_action = None
    state = gstate_dec.GroundState(graph=graph, goalID=goal.id, robot=robot, n_maps=sampling_maps)
    
    best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iter, rollout_fn= core.sctp_rollout3)
    print(best_action)
    print(cost)
    print([[a.target, a.start_pose] for a in path_cost[0]])  
    print([c for c in path_cost[1]])  

def test_groundstate_policy_mgraph():
    print()
    C=200.0
    num_iter=5000
    sampling_maps=500    
    start, goal, graph = graphs.m_graph_unc()
    robot = Robot(position=[-3.0, 4.0], cur_node=start.id, at_node=True, robot_type=param.RobotType.Ground)
    robot.unfinished_action = None
    state = gstate_dec.GroundState(graph=graph, goalID=goal.id, robot=robot, n_maps=sampling_maps)
    
    best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iter, rollout_fn= core.sctp_rollout3)
    print(best_action)
    print(cost)
    print([[a.target, a.start_pose] for a in path_cost[0]])  
    print([c for c in path_cost[1]])  

# def test_dronestate_policy_sgraph_2drones():
#     print()
#     C=200.0
#     num_iterations=500
    
#     start, goal, l_graph = graphs.s_graph_unc()
#     drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=param.RobotType.Drone) for _ in range(2)]
#     for drone in drones:
#         drone.unfinished_action = None
#     actions = [core.Action(target=v.id, rtype=param.RobotType.Drone,  start_pose = (0.0,0.0)) for v in l_graph.pois if v.id != start.id]
#     robot_edges = [[1,1]]
#     robot_pos = [[0.0,0.0]]
#     state = dstate_dec.DronesState(actions=actions, robot_pos=robot_pos, redges=robot_edges, \
#                             graph=l_graph, restID=goal.id, drones=drones)
    
#     best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iterations, rollout_fn= dstate_dec.drone_rollout)
#     print(best_action)
#     print(cost)
#     print([[a.target, a.start_pose] for a in path_cost[0]])  
#     print([c for c in path_cost[1]])  


# def test_dronestate_policy_sgraph_3drones():
#     print()
#     C=200.0
#     num_iterations=500
#     num_drones=3
    
#     start, goal, l_graph = graphs.s_graph_unc()
#     drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=param.RobotType.Drone) for _ in range(num_drones)]
#     for drone in drones:
#         drone.unfinished_action = None
#     actions = [core.Action(target=v.id, rtype=param.RobotType.Drone,  start_pose = (0.0,0.0)) for v in l_graph.pois if v.id != start.id]
#     robot_edges = [[1,1]]
#     robot_pos = [[0.0,0.0]]
#     state = dstate_dec.DronesState(actions=actions, robot_pos=robot_pos, redges=robot_edges, \
#                             graph=l_graph, restID=goal.id, drones=drones)
    
#     best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iterations, rollout_fn= dstate_dec.drone_rollout)
#     print(best_action)
#     print(cost)
#     print([[a.target, a.start_pose] for a in path_cost[0]])  
#     print([c for c in path_cost[1]])  


# def test_dronestate_policy_mgraph_3drones():
#     print()
#     C=200.0
#     num_iterations=1000000
#     num_drones=1
    
#     start, goal, graph = graphs.m_graph_unc()
#     drones = [Robot(position=[-3.0, 4.0], cur_node=start.id, robot_type=param.RobotType.Drone) for _ in range(num_drones)]
#     for drone in drones:
#         drone.unfinished_action = None
#     actions = [core.Action(target=v.id, rtype=param.RobotType.Drone,  start_pose = (-3.0,4.0)) for v in graph.pois if v.id != start.id]
#     print(f"The number of actions {len(actions)}")
#     robot_edges = [[1,2]]
#     robot_pos = [[-3.0,4.0]]
#     state = dstate_dec.DronesState(actions=actions, robot_pos=robot_pos, redges=robot_edges, \
#                             graph=graph, restID=goal.id, drones=drones)
    
#     best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iterations, rollout_fn= dstate_dec.drone_rollout)
#     print(best_action)
#     print(cost)
#     print([[a.target, a.start_pose] for a in path_cost[0]])  
#     print([c for c in path_cost[1]])  

