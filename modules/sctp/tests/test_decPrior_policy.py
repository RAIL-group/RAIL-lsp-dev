import pytest
import argparse
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.robot import Robot
from pouct_planner import core as policy
from sctp import param, dec_prior

def test_decPrior_policy_lgraph():
    print()
    C=200.0
    num_iterations=10
    
    starts, goals, l_graph = graphs.linear_graph_unc()
    uavs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True, robot_type=param.RobotType.Drone) for i in range(1)]
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(1)]
    
    for drone in uavs:
        drone.unfinished_action = None
    state = dec_prior.StateDecPrior(graph=l_graph, goalIDs=[goal.id for goal in goals], drones=uavs, ugvs=ugvs)
    
    best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iterations, rollout_fn= dec_prior.decsctp_rollout)
    print(best_action)
    print(cost)
    print([[a.target, a.start_pose] for a in path_cost[0]])  
    print([c for c in path_cost[1]])  


def test_decPrior_policy_dgraph():
    print()
    C=200.0
    num_iterations=1000
    uav_num = 1
    ugv_num = 1
    starts, goals, graph = graphs.disjoint_unc()
    uavs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True, robot_type=param.RobotType.Drone) for i in range(uav_num)]
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(ugv_num)]
    
    
    for drone in uavs:
        drone.unfinished_action = None
    state = dec_prior.StateDecPrior(graph=graph, goalIDs=[goal.id for goal in goals], drones=uavs, ugvs=ugvs)
    
    best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iterations, rollout_fn= dec_prior.decsctp_rollout)
    print(best_action)
    print(cost)
    print([[a.target, a.start_pose] for a in path_cost[0]])  
    print([c for c in path_cost[1]])  

def test_decPrior_policy_sgraph2goals():
    print()
    C=200.0
    num_iterations=1000
    uav_num = 2
    ugv_num = 2
    starts, goals, graph = graphs.s_graph_2goals()
    uavs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True, robot_type=param.RobotType.Drone) for i in range(uav_num)]
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(ugv_num)]
    
    
    for drone in uavs:
        drone.unfinished_action = None
    state = dec_prior.StateDecPrior(graph=graph, goalIDs=[goal.id for goal in goals], drones=uavs, ugvs=ugvs)
    
    best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iterations, rollout_fn= dec_prior.decsctp_rollout)
    print(best_action)
    print(cost)
    print([[a.target, a.start_pose] for a in path_cost[0]])  
    print([c for c in path_cost[1]])  


def test_decPrior_policy_mgraph3goals():
    print()
    C=200.0
    num_iterations=1000
    uav_num = 3
    ugv_num = 3
    starts, goals, graph = graphs.m_graph_unc()
    uavs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True, robot_type=param.RobotType.Drone) for i in range(uav_num)]
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(ugv_num)]
    
    
    for drone in uavs:
        drone.unfinished_action = None
    state = dec_prior.StateDecPrior(graph=graph, goalIDs=[goal.id for goal in goals], drones=uavs, ugvs=ugvs)
    
    best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iterations, rollout_fn= dec_prior.decsctp_rollout)
    print(best_action)
    print(cost)
    print([[a.target, a.start_pose] for a in path_cost[0]])  
    print([c for c in path_cost[1]])  
