import pytest
import argparse
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.robot import Robot
from pouct_planner import core as policy
from sctp import param, gstate_dec, dec_prior
import random
import numpy as np
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

def test_groundstate_policy_bridges_graph():
    print()
    seed = 3000
    random.seed(seed)
    np.random.seed(seed)
    C=200.0
    num_iterations=10000
    start, goal, graph = graphs.random_bridges_graph()
    # plotGraph = graph.copy()
    # policyGraph = graph.copy()
    print(f"The start position: {start.coord}, goal position: {goal.coord}")
    print(f"Node 9 {(graph.pois[0].id)} ' s position: {graph.pois[0].coord}")
    for poi in graph.pois:
        if poi.id in [9,11,12,13,14,15,16,17]:
            poi.block_prob = 0.0
        else:
            poi.block_prob = 1.0
    
    robot = Robot(position=[start.coord[0],start.coord[1]], cur_node=start.id, at_node=True, robot_type=param.RobotType.Ground)
    # sctpstate = gstate_dec.GroundState(graph=graph, goalID=goal.id, robot=robot, \
                            # useOptHeur=True, n_maps=80, revisit_pen=20.0)
    sctpstate = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, iscopy=False, n_maps=100)
    assert sctpstate.history.get_action_outcome(core.Action(target=10, start_pose=(start.coord[0],start.coord[1]))) == param.EventOutcome.BLOCK
    assert sctpstate.history.get_action_outcome(core.Action(target=12, start_pose=(start.coord[0],start.coord[1]))) == param.EventOutcome.TRAV
    assert sctpstate.history.get_action_outcome(core.Action(target=13, start_pose=(start.coord[0],start.coord[1]))) == param.EventOutcome.TRAV
    assert sctpstate.history.get_action_outcome(core.Action(target=14, start_pose=(start.coord[0],start.coord[1]))) == param.EventOutcome.TRAV
    assert sctpstate.history.get_action_outcome(core.Action(target=9, start_pose=(start.coord[0],start.coord[1]))) == param.EventOutcome.TRAV
    assert sctpstate.history.get_action_outcome(core.Action(target=15, start_pose=(start.coord[0],start.coord[1]))) == param.EventOutcome.TRAV
    assert sctpstate.history.get_action_outcome(core.Action(target=16, start_pose=(start.coord[0],start.coord[1]))) == param.EventOutcome.TRAV
    assert sctpstate.history.get_action_outcome(core.Action(target=17, start_pose=(start.coord[0],start.coord[1]))) == param.EventOutcome.TRAV
    best_action, cost, paths_costs, _, _  = policy.po_mcts(sctpstate, C=C, n_iterations=num_iterations, rollout_fn= core.sctp_rollout3)
    print(best_action)
    print(cost)
    # print([[a.target, a.start_pose] for a in path_cost[0]])  
    # print([c for c in path_cost[1]])  


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

