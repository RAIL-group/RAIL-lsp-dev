import numpy as np
import random
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.robot import Robot
from sctp.utils import plotting
from pouct_planner import core as policy
from sctp import param, jsap

def test_jsap_policy_lgraph():
    print()
    C=200.0
    num_iterations=100
    baseline = False
    useAVP = True
    
    starts, goals, l_graph = graphs.linear_graph_unc()
    if baseline:
        uavs = []
    else:
        uavs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True, robot_type=param.RobotType.Drone) for i in range(1)]
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(1)]
    
    for drone in uavs:
        drone.unfinished_action = None
    state = jsap.JSAPState(graph=l_graph, goalIDs=[goal.id for goal in goals], drones=uavs, ugvs=ugvs, \
                           useAVP=useAVP, max_uanum=1, n_maps=80)
    
    best_action, cost, path_cost_times  = policy.po_mcts(state, C=C, n_iterations=num_iterations, rollout_fn= jsap.decsctp_rollout)
    print(best_action)
    print(cost)
    print([[a.target, a.start_pose] for a in path_cost_times[0]])  
    print([c for c in path_cost_times[1]])  


def test_jsap_policy_dgraph():
    print()
    C=200.0
    num_iterations=2500
    uav_num = 1
    ugv_num = 1
    starts, goals, graph = graphs.disjoint_unc()
    baseline = False
    useAVP = True
    planner = 'JSAP-AVP'
    if baseline:
        uavs = []
    else:
        uavs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True, robot_type=param.RobotType.Drone) for i in range(uav_num)]
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(ugv_num)]
    
    
    for drone in uavs:
        drone.unfinished_action = None
    state = jsap.JSAPState(graph=graph, goalIDs=[goal.id for goal in goals], drones=uavs, \
                    useAVP=useAVP, max_uanum=1, ugvs=ugvs, n_maps=80)
    
    best_action, cost, path_cost_times  = policy.po_mcts(state, C=C, n_iterations=num_iterations, rollout_fn= jsap.decsctp_rollout)
    print("The list of actions: ", [f"{a}" for a in path_cost_times[0]])  
    print("The cost are: ", [f"{c:.2f}"  for c in path_cost_times[1]])
    if useAVP:
        print(f"The sampling time: {path_cost_times[2]:.2f} seconds using {planner} ")
    
    plotting.plot_policy(graph, actions=path_cost_times[0], name=f"{planner} Policy", startID=starts[0].id, \
                               goalID=goals[0].id, seed=2000, verbose=True)

def test_jsap_policy_sgraph_1uav1ugv():
    print()
    seed = 2000
    np.random.seed(seed)
    random.seed(seed)
    C=200.0
    num_iterations=1500
    uav_num = 1
    ugv_num = 1
    starts, goals, graph = graphs.s_graph_unc()
    baseline = False 
    useAVP = True
    planner = 'JSAP-AVP'
    if baseline:
        uavs = []
    else:
        uavs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True, robot_type=param.RobotType.Drone) for i in range(uav_num)]
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(ugv_num)]
    
    
    for drone in uavs:
        drone.unfinished_action = None
    state = jsap.JSAPState(graph=graph, goalIDs=[goal.id for goal in goals], drones=uavs, ugvs=ugvs, \
                           useAVP=useAVP, max_uanum=1, n_maps=80)
    
    best_action, cost, path_cost_times  = policy.po_mcts(state, C=C, n_iterations=num_iterations, rollout_fn= jsap.decsctp_rollout)
    
    
    print("The list of actions: ", [[f"{a.target}", f"({a.start_pose[0]:.2f}, {a.start_pose[1]:.2f})"] for a in path_cost_times[0]])  
    print("The cost are: ", [f"{c:.2f}"  for c in path_cost_times[1]])  
    if useAVP:
        assert path_cost_times[2] > 0.0
        print(f"The sampling time: {path_cost_times[2]:.2f} seconds using {planner} ")
    else:
        assert path_cost_times[2] == 0.0
    plotting.plot_policy(graph, actions=path_cost_times[0], name=f"{planner} Policy", startID=starts[0].id, \
                               goalID=goals[0].id, seed=seed, verbose=True)


def test_jsap_policy_sgraph2goals():
    print()
    seed = 2000
    C=200.0
    num_iterations=2000
    uav_num = 0
    ugv_num = 2
    starts, goals, graph = graphs.s_graph_2goals()
    baseline = True
    useAVP = True
    
    if baseline:
        uavs = []
        planner = 'CTP'
        mdepth = 15
        useAVP = False
    else:
        mdepth = 8
        if useAVP:
            planner = 'JSAP-AVP'
        else:
            planner = 'JSAP'
        uavs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True, robot_type=param.RobotType.Drone) for i in range(uav_num)]
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(ugv_num)]
    
    
    for drone in uavs:
        drone.unfinished_action = None
    state = jsap.JSAPState(graph=graph, goalIDs=[goal.id for goal in goals], drones=uavs, ugvs=ugvs, 
                           useAVP=useAVP, max_uanum=1, n_maps=60)
    
    best_action, cost, path_cost_times  = policy.po_mcts(state, C=C, n_iterations=num_iterations, \
                            depth=mdepth, rollout_fn= jsap.decsctp_rollout)
    
    print("The list of actions: ", [[f"{a.target}", f"({a.start_pose[0]:.2f}, {a.start_pose[1]:.2f})"] for a in path_cost_times[0]])  
    print("The cost are: ", [f"{c:.2f}"  for c in path_cost_times[1]])  
    
    if useAVP:
        assert path_cost_times[2] > 0.0
        print(f"The sampling time: {path_cost_times[2]:.2f} seconds using {planner} ")
    else:
        assert path_cost_times[2] == 0.0
    
    plotting.plot_policy(graph, actions=path_cost_times[0], startID=starts[0].id, \
                               goalID=goals[0].id, seed=seed, verbose=True)


    
def test_jsap_policy_sgraph_2uavs2ugvs():
    print()
    C=200.0
    num_iterations=800
    # uav_num = 2
    ugv_num = 2
    starts, goals, graph = graphs.s_graph_2goals()
    
    for poi in graph.pois:
        if poi.id == 9 or poi.id==6 or poi.id==10:
            poi.block_status = 0
        elif poi.id == 11:
            poi.block_status = 1
            
    uav1 = Robot(position=[7.76, 1.94], cur_node=1, at_node=False, robot_type=param.RobotType.Drone)
    uav2 = Robot(position=[4.0, 0.0], cur_node=3, at_node=True, robot_type=param.RobotType.Drone)
    baseline = True
    if baseline:
        uavs = []
    else:
        uavs = [uav1, uav2]
    ugvs = [Robot(position=[1.886, 1.886], cur_node=1, at_node=False, edge=[1,6]) for i in range(ugv_num)]
    use_AVP = False 
    max_uanum = 1
    
    for drone in uavs:
        drone.unfinished_action = None
    state = jsap.JSAPState(graph=graph, goalIDs=[goal.id for goal in goals], drones=uavs, ugvs=ugvs,
                                    useAVP=use_AVP, max_uanum=max_uanum, n_maps=80)
    
    best_action, cost, path_cost_times  = policy.po_mcts(state, C=C, \
        n_iterations=num_iterations, rollout_fn= jsap.decsctp_rollout)
    
    # print("The best action: ", best_action)
    # print(f"With cost: {cost:.2f}")
    # print("The sampling time: ", f"{path_cost_times[2]:.2f} seconds")
    print("The list of actions: ", [[a.target, a.start_pose] for a in path_cost_times[0]])  
    print("The cost are: ", [f"{c:.2f}"  for c in path_cost_times[1]])  


def test_jsap_policy_mgraph3goals():
    print()
    C=200.0
    num_iterations=2000
    uav_num = 1
    ugv_num = 1
    starts, goals, graph = graphs.m_graph_unc()
    baseline = True
    useAVP = False
    max_uanum = 1
    if baseline:
        uavs = []
        planner = 'CTP'
        mdepth = 20
    else:
        mdepth = 9
        if useAVP:
            planner = 'JSAP-AVP'
        else:
            planner = 'JSAP'
        uavs = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, at_node=True, \
                    robot_type=param.RobotType.Drone) for i in range(uav_num)]
    ugvs = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, at_node=True) \
                    for i in range(ugv_num)]
    
    for drone in uavs:
        drone.unfinished_action = None
    state = jsap.JSAPState(graph=graph, goalIDs=[goal.id for goal in goals], drones=uavs, \
                            ugvs=ugvs, useAVP=useAVP, max_uanum=max_uanum, n_maps=60)
    best_action, cost, path_cost_times  = policy.po_mcts(state, C=C, \
                            n_iterations=num_iterations, depth=mdepth, rollout_fn= jsap.decsctp_rollout)
    
    print("The list of actions: ", [[f"{a.target}", f"({a.start_pose[0]:.2f}, {a.start_pose[1]:.2f})"] for a in path_cost_times[0]])
    print("The cost are: ", [f"{c:.2f}"  for c in path_cost_times[1]])  
    if useAVP:
        assert path_cost_times[2] > 0.0
        print(f"The sampling time: {path_cost_times[2]:.2f} seconds using {planner} ")
    else:
        assert path_cost_times[2] == 0.0
    
    plotting.plot_policy(graph, actions=path_cost_times[0], name=f"{planner} Policy", startID=starts[0].id, \
                               goalID=goals[0].id, seed=2000, verbose=True)

