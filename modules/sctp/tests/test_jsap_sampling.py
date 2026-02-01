import numpy as np
import random, time
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.robot import Robot
from sctp.utils import plotting
from pouct_planner import core as policy
from sctp import param, jsap
from sctp.scripts import data_gen 


def test_sap_sampling_sgraph():
    print()
    seed = 2000
    np.random.seed(seed)
    random.seed(seed)
    C=200.0
    # num_iterations=1500
    uav_num = 1
    ugv_num = 1
    starts, goals, graph = graphs.get_insland_bridges_graph() # graphs.s_graph_unc()
    baseline = False 
    useAVP = True
    planner = 'SAP-IAP'
    if baseline:
        uavs = []
    else:
        uavs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True, robot_type=param.RobotType.Drone) for i in range(uav_num)]
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(ugv_num)]
    
    
    for drone in uavs:
        drone.unfinished_action = None
    state = jsap.JSAPState(graph=graph, goalIDs=[goal.id for goal in goals], drones=uavs, ugvs=ugvs, \
                           useAVP=useAVP, max_uanum=1, n_maps=200)
    
    time1 = time.perf_counter()
    state.update_action_bc_networkX()
    for action, bc in state.behavior_change.items():
        print(f"Action to {action.target} has BC value: {bc:.2f}")
    networkX_time = time.perf_counter() - time1
    print("---------------------------------------------------------")
    
    time1 = time.perf_counter()
    state.update_action_bc()
    for action, bc in state.behavior_change.items():
        print(f"Action to {action.target} has BC value: {bc:.2f}")
    old_time = time.perf_counter() - time1
    
    print (f"NetworkX time: {networkX_time:.2f} seconds, Old time: {old_time:.2f} seconds")
    plotting.plot_policy(graph, actions=[], name=f"{planner} Policy", startID=[starts[0].id], \
                               goalID=[goals[0].id], seed=seed, verbose=True)


def test_jsap_gen_data():
    path = '/data/sctp/jsap_sgraph_data'
    num_graphs = 5
    num_maps = 60
    data_gen.generate_dataset(path=path, num_graphs=num_graphs,num_maps=num_maps)
 