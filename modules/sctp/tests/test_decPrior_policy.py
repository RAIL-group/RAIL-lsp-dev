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
    
def test_decPrior_policy_sgraph_2uavs2ugvs():
    print()
    C=200.0
    num_iterations=800
    uav_num = 2
    ugv_num = 2
    starts, goals, graph = graphs.s_graph_2goals()
    
    for poi in graph.pois:
        if poi.id == 9 or poi.id==6 or poi.id==10:
            poi.block_status = 0
        elif poi.id == 11:
            poi.block_status = 1
            
    uav1 = Robot(position=[7.76, 1.94], cur_node=1, at_node=False, robot_type=param.RobotType.Drone)
    uav2 = Robot(position=[8.0, 0.0], cur_node=4, at_node=True, robot_type=param.RobotType.Drone)
    uavs = [uav1, uav2]
    ugvs = [Robot(position=[1.886, 1.886], cur_node=1, at_node=False, edge=[1,6]) for i in range(ugv_num)]
    use_2AG = True 
    max_uanum = 1
    
    for drone in uavs:
        drone.unfinished_action = None
    state = dec_prior.StateDecPrior(graph=graph, goalIDs=[goal.id for goal in goals], drones=uavs, ugvs=ugvs,
                                    use_2AG=use_2AG, max_uanum=max_uanum)
    
    best_action, cost, path_cost  = policy.po_mcts(state, C=C, n_iterations=num_iterations, rollout_fn= dec_prior.decsctp_rollout)
    print(best_action)
    print(cost)
    print([[a.target, a.start_pose] for a in path_cost[0]])  
    print([c for c in path_cost[1]])  


def test_decPrior_policy_mgraph3goals():
    print()
    C=200.0
    num_iterations=1500
    uav_num = 1
    ugv_num = 1
    useAVP = True
    max_uanum = 1
    starts, goals, graph = graphs.m_graph_unc()
    uavs = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, at_node=True, \
                    robot_type=param.RobotType.Drone) for i in range(uav_num)]
    
    ugvs = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, at_node=True) \
                    for i in range(ugv_num)]
    
    # uavs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True, robot_type=param.RobotType.Drone) for i in range(uav_num)]
    # ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(ugv_num)]
    
    
    for drone in uavs:
        drone.unfinished_action = None
    state = dec_prior.StateDecPrior(graph=graph, goalIDs=[goal.id for goal in goals], drones=uavs, ugvs=ugvs,
                                    n_maps=80, use2AG=useAVP, max_uanum=max_uanum, spolicy_rollouts=300)
    
    best_action, cost, path_cost, sampling_time, spolicy_time  = policy.po_mcts(state, C=C, \
            n_iterations=num_iterations, depth=30, rollout_fn= dec_prior.decsctp_rollout)
    
    
    print("The best action: ", best_action)
    print(f"With cost: {cost:.2f}")
    print("The sampling time: ", f"{sampling_time:.2f} seconds")
    print("The list of actions: ", [[a.target, a.start_pose] for a in path_cost[0]])  
    print("The cost are: ", [f"{c:.2f}"  for c in path_cost[1]])  