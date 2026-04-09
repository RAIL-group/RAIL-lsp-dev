import random, time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sctp import sctp_graphs as graphs
from sctp.utils import plotting 
from sctp.robot import Robot
from sctp import jsap, param
from sctp.param import RobotType
from sctp.planners import jsap_planner as planner
from sctp.planners import jsap_plan_exe as plan_loop


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='base')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_ugvs', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--C', type=int, default=300)
    parser.add_argument('--max_depth', type=int, default=500)
    parser.add_argument('--n_maps', type=int, default=80)
    parser.add_argument('--n_vertex', type=int, default=16)

    args = parser.parse_args(['--save_dir', ''])
    
    args.save_dir = 'data/sctp'
    args.planner = 'jsapliap'
    # args.num_drones = 
    args.num_iterations = 500
    args.C = 200
    args.max_depth = 30
    args.current_seed = args.seed
    args.num_ugvs = 1
    
    return args


def test_jsap_plan_exec_lg():
    print()
    args = _get_args()
    # args.planner = 'jsap'
    random.seed(args.seed)
    np.random.seed(args.seed)

    starts, goals, graph = graphs.linear_graph_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        poi.block_status = 0
    
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    use_AVP = False 
    max_uanum = 1
    if args.planner == 'ctp':
        drones = []
    else:
        drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, robot_type=RobotType.Drone, at_node=True)]
    robots = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, at_node=True)]
    
    planner_robots = [robot.copy() for robot in robots]
    planner_drones = [drone.copy() for drone in drones]
    jsapplanner = planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, 
                            uavs=planner_drones, rollout_fn=jsap.decsctp_rollout, C=args.C, revisit_pen=param.REVISIT_PEN,\
                            rollout_num=args.num_iterations, tree_depth=args.max_depth, n_maps=args.n_maps, use_AVP=use_AVP,
                            max_uanum=max_uanum, verbose=True)
    plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=jsapplanner.reached_goal)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    
    for step_data in plan_exec:
        jsapplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_action, cost = jsapplanner.compute_joint_action()
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        plan_exec.save_joint_actions(joint_action, cost)
    
    cost_sum = np.sum([robot.net_time for robot in robots])
    cost_aver = np.average([robot.net_time for robot in robots])

    runtime = time.perf_counter() - start_time
    average_step_time /= count_steps
    gpaths = []
    for robot in robots:
        x_g = [pose[0] for pose in robot.all_poses]
        y_g = [pose[1] for pose in robot.all_poses]
        gpaths.append([x_g, y_g])
    
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    goals_cords = [goal.coord for goal in goals]
    starts_cords = [start.coord for start in starts]
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpaths=gpaths, dpaths=dpaths, \
                    graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
                        seed=args.seed, ttime=runtime, stime=average_step_time, cost=cost_aver, verbose=True)
    plt.show()
    

def test_jsap_plan_exec_dg():
    print()
    args = _get_args()
    args.planner = 'jsapiap'
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.num_iterations = 500
    args.n_maps = 100

    starts, goals, graph = graphs.disjoint_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        if poi.id == 8 or poi.id==6 or poi.id==5:
            poi.block_status = 0
        else:
            poi.block_status = 1
    
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    use_AVP = False 
    max_uanum = 1
    robots = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, at_node=True)]
    if args.planner == 'ctp':
        drones = []
    else:
        drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, robot_type=RobotType.Drone, at_node=True)]
    planner_robots = [robot.copy() for robot in robots] 
    planner_drones = [drone.copy() for drone in drones]
    jsapplanner = planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, uavs=planner_drones,
                                            rollout_fn=jsap.decsctp_rollout, C=args.C, rollout_num=args.num_iterations,
                                            tree_depth=args.max_depth, n_maps=args.n_maps, use_AVP=use_AVP,
                                            max_uanum=max_uanum, verbose=True)
    plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=jsapplanner.reached_goal)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    
    for step_data in plan_exec:
        jsapplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_action, cost = jsapplanner.compute_joint_action()
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        plan_exec.save_joint_actions(joint_action, cost)
    
    cost_sum = np.sum([robot.net_time for robot in robots])
    cost_aver = np.average([robot.net_time for robot in robots])

    runtime = time.perf_counter() - start_time
    average_step_time /= count_steps
    gpaths = []
    for robot in robots:
        x_g = [pose[0] for pose in robot.all_poses]
        y_g = [pose[1] for pose in robot.all_poses]
        gpaths.append([x_g, y_g])
    
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    goals_cords = [goal.coord for goal in goals]
    starts_cords = [start.coord for start in starts]
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpaths=gpaths, dpaths=dpaths, \
                    graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
                        seed=args.seed, cost=cost_aver, ttime=runtime, stime=average_step_time, verbose=False)
    plt.show()
    print("Done Plotting")
    

def test_jsap_plan_exec_mgraph():
    print()
    args = _get_args()
    args.planner = 'jsapiap'
    args.num_iterations = 500
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.num_drones = 1
    args.n_maps = 150
    

    starts, goals, graph = graphs.m_graph_unc()
    num_uav = 1
    num_ugv = 1
    
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    use_AVP = True 
    max_uanum = 1
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(num_ugv)]
    if args.planner == 'ctp':
        drones = []
        args.num_iterations = 1000
        args.max_depth = 15
    else:
        args.max_depth = 8
        if use_AVP:
            max_uanum = num_uav
            args.num_iterations = 1000
            args.planner = 'jsapiap'
        else:
            max_uanum = 1
            args.num_iterations = 1000
            args.planner = 'jsap'
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(num_uav)]
    planner_robots = [robot.copy() for robot in robots]
    planner_drones = [drone.copy() for drone in drones]
    jsapplanner = planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, uavs=planner_drones,
                                            rollout_fn=jsap.decsctp_rollout, C=args.C, rollout_num=args.num_iterations,
                                            tree_depth=args.max_depth, n_maps=args.n_maps, use_AVP=use_AVP,
                                            max_uanum=max_uanum, verbose=True)
    plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=jsapplanner.reached_goal)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    
    for step_data in plan_exec:
        jsapplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_action, cost = jsapplanner.compute_joint_action()
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        plan_exec.save_joint_actions(joint_action, cost)
    
    cost_sum = np.sum([robot.net_time for robot in robots])
    cost_aver = np.average([robot.net_time for robot in robots])

    runtime = time.perf_counter() - start_time
    average_step_time /= count_steps    
    gpaths = []
    for robot in robots:
        x_g = [pose[0] for pose in robot.all_poses]
        y_g = [pose[1] for pose in robot.all_poses]
        gpaths.append([x_g, y_g])
    
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    goals_cords = [goal.coord for goal in goals]
    starts_cords = [start.coord for start in starts]
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpaths=gpaths, dpaths=dpaths, \
                    graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
                        seed=args.seed, cost=cost_sum, ttime=runtime, stime=average_step_time, verbose=False)
    plt.show()


def test_jsap_plan_exec_randomgraph():
    print()
    args = _get_args()
    args.planner = 'jsapdap'
    args.seed = 3022
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.num_ugvs =2
    args.num_drones =1
    verbose = True
    args.n_vertex = 16
    # args.num_ugvs = 2
    env_type = 'random'
    if env_type == 'bridges':
       starts, goals, graph = graphs.get_bridges_graph()
    elif env_type == 'random': 
        starts, goals, graph = graphs.random_graph(n_vertex=args.n_vertex, SG_pairs=3)
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    
    if args.planner == 'ctp':
        args.num_drones =0
        drones = []
        args.num_iterations = 1500 #2000 
        args.max_depth = 12
        use_AVP = False
        use_DAP = False
        param.REVISIT_PEN = 0.0
        max_uanum = 1
    elif args.planner == 'jsap':
        use_AVP = False
        use_DAP = False
        args.max_depth = 18
        args.num_drones =1 
        # args.num_ugvs =2
        max_uanum = 1
        args.num_iterations = 1000
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
    elif args.planner == 'jsap2':
        use_AVP = False
        use_DAP = False
        args.num_drones = 2
        assert args.num_ugvs > 0
        args.max_depth = 18
        param.REVISIT_PEN = 0.0
        assert args.num_drones == 2
        assert args.num_ugvs == 3 
        max_uanum = 1
        args.num_iterations = 30000 # 6000+3500*(args.num_drones-1)
        drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, \
                robot_type=RobotType.Drone, at_node=True) for _ in range(args.num_drones)]

    elif args.planner == 'jsapiap':
        use_AVP=True
        use_DAP = False
        assert args.num_drones == 1
        assert args.num_ugvs == 1
        args.max_depth = 15
        args.num_iterations = 1000
        args.n_maps = 200
        max_uanum = 1
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        print(f"Testing JSAP-IAP planner with use_IAP={use_AVP} and num_iterations={args.num_iterations} and max_depth={args.max_depth}")
    
    elif args.planner == 'jsapdap':
        use_AVP=False
        use_DAP = True
        use_Learning=False
        model_path =""
        assert args.num_drones == 1
        assert args.num_ugvs == 2
        args.max_depth = 15
        # max_uanum = max_uanum
        args.num_iterations = 1000
        args.n_maps = 200
        max_uanum = 1
        
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        print(f"Testing JSAP-DAP planner with use_DAP={use_DAP} and num_iterations={args.num_iterations} and max_depth={args.max_depth}")
    else:
        raise ValueError(f'Planner {args.planner} not recognized')
    
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(args.num_ugvs)]
    planner_robots = [robot.copy() for robot in robots]
    planner_drones = [drone.copy() for drone in drones]
    
    # poi17 = graph.get_poi(17)
    # print(f"POI17 has {len(poi17.neighbors)} neighbors: {poi17.neighbors}")
    # jsapplanner = planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, \
    #             uavs=planner_drones, rollout_fn=jsap.decsctp_rollout, C=args.C, revisit_pen=param.REVISIT_PEN, \
    #             rollout_num=args.num_iterations, tree_depth=args.max_depth, n_maps=args.n_maps, use_DAP=use_DAP, \
    #             use_AVP=use_AVP, max_uanum=max_uanum, verbose=True)
    jsapplanner = planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, 
                                              uavs=planner_drones, rollout_fn=jsap.decsctp_rollout, C=args.C, 
                                              rollout_num=args.num_iterations, tree_depth=args.max_depth, n_maps=args.n_maps, 
                                              use_AVP=use_AVP, use_DAP=use_DAP, useLearning=use_Learning, model_path=model_path,\
                                              max_uanum=max_uanum, verbose=True)
    
    plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=jsapplanner.reached_goal, verbose=True)

    goals_cords = [goal.coord for goal in goals]
    starts_cords = [start.coord for start in starts]
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpaths=[[[0.0]]], dpaths=[[[0.0]]], \
                    graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
                        seed=args.seed, cost=0.0, ttime=0.0, stime=0.0, verbose=verbose)
    plt.savefig(f'{args.save_dir}/figures/sctp_eval_planner_{args.planner}_seed_{args.seed}_{args.num_drones}UAVs_Test.pdf')    

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    print(f"{args.planner}: working on a team of {len(robots)} UGVs - {len(drones)} UAVs, seed {args.seed}, iter. {args.num_iterations}")
    
    for step_data in plan_exec:
        jsapplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_action, cost = jsapplanner.compute_joint_action()
        # break
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        plan_exec.save_joint_actions(joint_action, cost)
        
    print(f"The sampling time: {jsap.JSAPState.total_sampling_time}s")
    robots_net_times = [robot.net_time for robot in robots]
    
    cost_sum = np.sum(robots_net_times)
    cost_aver = np.average(robots_net_times)

    runtime = time.perf_counter() - start_time
    # average_step_time /= count_steps
    print(f"The total time: {runtime}") 
    average_step_time = 0.0    
    gpaths = []
    for robot in robots:
        x_g = [pose[0] for pose in robot.all_poses]
        y_g = [pose[1] for pose in robot.all_poses]
        gpaths.append([x_g, y_g])
    
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    # goals_cords = [goal.coord for goal in goals]
    # starts_cords = [start.coord for start in starts]
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpaths=gpaths, dpaths=dpaths, \
                    graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
                        seed=args.seed, cost=cost_sum, ttime=runtime, stime=average_step_time, verbose=verbose)
    plt.savefig(f'{args.save_dir}/figures/sctp_eval_planner_{args.planner}_seed_{args.seed}_{args.num_drones}UAVs.pdf')    
    plt.show()



def test_jsap_plan_exec_bridges_graph():
    print()
    args = _get_args()
    args.planner = 'jsapliap'
    args.seed = 3001
    random.seed(args.seed)
    np.random.seed(args.seed)
    verbose = True
    args.num_ugvs =1
    starts, goals, graph = graphs.get_bridges_graph()
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    
    num_ugvs = 1
    
    if args.planner == 'ctp':
        args.num_drones =0
        drones = []
        args.num_iterations = 1000 #2000 
        args.max_depth = 12
        use_AVP = False
        use_DAP = False
        param.REVISIT_PEN = 0.0
        max_uanum = 1
    elif args.planner == 'jsap':
        use_AVP = False
        use_DAP = False
        args.max_depth = 15
        args.num_drones =1 
        assert args.num_ugvs == num_ugvs
        max_uanum = 1
        args.num_iterations = 1000
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
    elif args.planner == 'jsap2':
        use_AVP = False
        use_DAP = False
        args.num_drones = 2
        assert args.num_ugvs > 0
        args.max_depth = 18
        param.REVISIT_PEN = 0.0
        assert args.num_drones == 2
        assert args.num_ugvs == 3 
        max_uanum = 1
        args.num_iterations = 1000 # 6000+3500*(args.num_drones-1)
        drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, \
                robot_type=RobotType.Drone, at_node=True) for _ in range(args.num_drones)]

    elif args.planner == 'jsapiap':
        use_AVP=True
        use_DAP = False
        assert args.num_drones > 0
        assert args.num_ugvs > 0
        args.max_depth = 15
        # max_uanum = max_uanum
        args.num_iterations = 1000
        args.n_maps = 200
        max_uanum = 1
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        print(f"Testing JSAP-IAP planner with use_IAP={use_AVP} and num_iterations={args.num_iterations} and max_depth={args.max_depth}")
    
    elif args.planner == 'jsapdap':
        use_AVP=False
        use_DAP = True
        assert args.num_drones > 0
        assert args.num_ugvs > 0
        args.max_depth = 15
        # max_uanum = max_uanum
        args.num_iterations = 1000
        args.n_maps = 200
        max_uanum = 1
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        print(f"Testing JSAP-DAP planner with use_DAP={use_DAP} and num_iterations={args.num_iterations} and max_depth={args.max_depth}")
    elif args.planner == 'jsapliap':
        use_AVP=False
        use_DAP = False
        use_Learning = True
        assert args.num_drones > 0
        assert args.num_ugvs > 0
        model_path ='modules/sctp/learning/models/iap_gnn_allgraphs_l.pt'
        args.max_depth = 15
        # max_uanum = max_uanum
        args.num_iterations = 1000
        args.n_maps = 200
        max_uanum = 1
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        print(f"Testing JSAP-IAP planner with use_IAP={use_AVP} and num_iterations={args.num_iterations} and max_depth={args.max_depth}")
        
    else:
        raise ValueError(f'Planner {args.planner} not recognized')
    
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(args.num_ugvs)]
    planner_robots = [robot.copy() for robot in robots]
    planner_drones = [drone.copy() for drone in drones]
    
    jsapplanner = planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, 
                                              uavs=planner_drones, rollout_fn=jsap.decsctp_rollout, C=args.C, 
                                              rollout_num=args.num_iterations, tree_depth=args.max_depth, n_maps=args.n_maps, 
                                              use_AVP=use_AVP, use_DAP=use_DAP, useLearning=use_Learning, model_path=model_path,\
                                              max_uanum=max_uanum, verbose=False)
    plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=jsapplanner.reached_goal, verbose=False)

    
    # jsapplanner = planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, \
    #             uavs=planner_drones, rollout_fn=jsap.decsctp_rollout, C=args.C, revisit_pen=param.REVISIT_PEN, \
    #             rollout_num=args.num_iterations, tree_depth=args.max_depth, n_maps=args.n_maps, use_DAP=use_DAP, \
    #             use_AVP=use_AVP, max_uanum=max_uanum, verbose=True)
    
    
    # plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
    #                                                 reached_goal=jsapplanner.reached_goal, verbose=True)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    print(f"{args.planner}: working on a team of {len(robots)} UGVs - {len(drones)} UAVs, seed {args.seed}, iter. {args.num_iterations}")
    
    for step_data in plan_exec:
        jsapplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_action, cost = jsapplanner.compute_joint_action()
        # break
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        plan_exec.save_joint_actions(joint_action, cost)
        
    print(f"The sampling time: {jsap.JSAPState.total_sampling_time}s")
    robots_net_times = [robot.net_time for robot in robots]
    
    cost_sum = np.sum(robots_net_times)
    cost_aver = np.average(robots_net_times)

    runtime = time.perf_counter() - start_time
    # average_step_time /= count_steps
    print(f"The total time: {runtime}") 
    average_step_time = 0.0    
    gpaths = []
    for robot in robots:
        x_g = [pose[0] for pose in robot.all_poses]
        y_g = [pose[1] for pose in robot.all_poses]
        gpaths.append([x_g, y_g])
    
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    goals_cords = [goal.coord for goal in goals]
    starts_cords = [start.coord for start in starts]
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpaths=gpaths, dpaths=dpaths, \
                    graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
                        seed=args.seed, cost=cost_sum, ttime=runtime, stime=average_step_time, verbose=verbose)
    plt.savefig(f'{args.save_dir}/figures/sctp_eval_planner_{args.planner}_seed_{args.seed}_{args.num_drones}UAVs.pdf')    
    plt.show()


def test_jsap_plan_exec_island_graph():
    print()
    args = _get_args()
    args.planner = 'ctp'
    args.seed = 3017
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.num_ugvs = 1
    # args.num_drones = 1
    verbose = False
    starts, goals, graph = graphs.get_sixIslands_graph()
    # print(f"The number of vertices in the graph: {len(graph.vertices)}")
    # print(f"The vertices' ID: {[v.id for v in graph.vertices]}")
    # return
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    
    if args.planner == 'ctp':
        args.num_drones =0
        drones = []
        args.num_iterations = 1000 #2000 
        args.max_depth = 13
        use_AVP = False
        use_DAP = False
        param.REVISIT_PEN = 0.0
        max_uanum = 1
    elif args.planner == 'jsap':
        use_AVP = False
        use_DAP = False
        args.max_depth = 15
        args.num_drones =1 
        # args.num_ugvs =2
        max_uanum = 1
        args.num_iterations = 1000
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
    elif args.planner == 'jsap2':
        use_AVP = False
        use_DAP = False
        assert args.num_drones == 2
        assert args.num_ugvs > 0
        args.max_depth = 15
        param.REVISIT_PEN = 0.0
        max_uanum = 1
        args.num_iterations = 1500 # 1000
        drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, \
                robot_type=RobotType.Drone, at_node=True) for _ in range(args.num_drones)]
        assert args.num_drones == 2
        assert args.num_ugvs == 2
    elif args.planner == 'jsapiap':
        use_AVP=True
        use_DAP = False
        args.num_drones = 1 
        assert args.num_ugvs > 0
        args.max_depth = 15
        # max_uanum = max_uanum
        args.num_iterations = 1000
        args.n_maps = 200
        max_uanum = 1
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        print(f"Testing JSAP-IAP planner with use_IAP={use_AVP} and num_iterations={args.num_iterations} and max_depth={args.max_depth}")
    
    elif args.planner == 'jsapdap':
        use_AVP=False
        use_DAP = True
        args.num_drones = 1
        assert args.num_ugvs > 0
        args.max_depth = 15
        # max_uanum = max_uanum
        args.num_iterations = 1000
        args.n_maps = 200
        max_uanum = 1
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        print(f"Testing JSAP-DAP planner with use_DAP={use_DAP} and num_iterations={args.num_iterations} and max_depth={args.max_depth}")
    else:
        raise ValueError(f'Planner {args.planner} not recognized')
    
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(args.num_ugvs)]
    planner_robots = [robot.copy() for robot in robots]
    planner_drones = [drone.copy() for drone in drones]
    jsapplanner = planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, \
                uavs=planner_drones, rollout_fn=jsap.decsctp_rollout, C=args.C, revisit_pen=param.REVISIT_PEN, \
                rollout_num=args.num_iterations, tree_depth=args.max_depth, n_maps=args.n_maps, use_DAP=use_DAP, \
                use_AVP=use_AVP, max_uanum=max_uanum, verbose=True)
    
    
    plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=jsapplanner.reached_goal, verbose=True)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    print(f"{args.planner}: working on a team of {len(robots)} UGVs - {len(drones)} UAVs, seed {args.seed}, iter. {args.num_iterations}")
    
    for step_data in plan_exec:
        jsapplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_action, cost = jsapplanner.compute_joint_action()
        # break
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        plan_exec.save_joint_actions(joint_action, cost)
        
    print(f"The sampling time: {jsap.JSAPState.total_sampling_time}s")
    robots_net_times = [robot.net_time for robot in robots]
    
    cost_sum = np.sum(robots_net_times)
    cost_aver = np.average(robots_net_times)

    runtime = time.perf_counter() - start_time
    # average_step_time /= count_steps
    print(f"The total time: {runtime}") 
    average_step_time = 0.0    
    gpaths = []
    for robot in robots:
        x_g = [pose[0] for pose in robot.all_poses]
        y_g = [pose[1] for pose in robot.all_poses]
        gpaths.append([x_g, y_g])
    
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    goals_cords = [goal.coord for goal in goals]
    starts_cords = [start.coord for start in starts]
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpaths=gpaths, dpaths=dpaths, \
                    graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
                        seed=args.seed, cost=cost_sum, ttime=runtime, stime=average_step_time, verbose=verbose)
    plt.savefig(f'{args.save_dir}/figures/sctp_eval_planner_{args.planner}_seed_{args.seed}_{args.num_drones}UAVs.pdf')    
    plt.show()



def test_jsap_plan_exec_island_graph():
    print()
    args = _get_args()
    args.planner = 'ctp'
    args.seed = 3017
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.num_ugvs = 1
    # args.num_drones = 1
    verbose = False
    starts, goals, graph = graphs.get_sixIslands_graph()
    # print(f"The number of vertices in the graph: {len(graph.vertices)}")
    # print(f"The vertices' ID: {[v.id for v in graph.vertices]}")
    # return
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    
    if args.planner == 'ctp':
        args.num_drones =0
        drones = []
        args.num_iterations = 1000 #2000 
        args.max_depth = 13
        use_AVP = False
        use_DAP = False
        param.REVISIT_PEN = 0.0
        max_uanum = 1
    elif args.planner == 'jsap':
        use_AVP = False
        use_DAP = False
        args.max_depth = 15
        args.num_drones =1 
        # args.num_ugvs =2
        max_uanum = 1
        args.num_iterations = 1000
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
    elif args.planner == 'jsap2':
        use_AVP = False
        use_DAP = False
        assert args.num_drones == 2
        assert args.num_ugvs > 0
        args.max_depth = 15
        param.REVISIT_PEN = 0.0
        max_uanum = 1
        args.num_iterations = 1500 # 1000
        drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, \
                robot_type=RobotType.Drone, at_node=True) for _ in range(args.num_drones)]
        assert args.num_drones == 2
        assert args.num_ugvs == 2
    elif args.planner == 'jsapiap':
        use_AVP=True
        use_DAP = False
        args.num_drones = 1 
        assert args.num_ugvs > 0
        args.max_depth = 15
        # max_uanum = max_uanum
        args.num_iterations = 1000
        args.n_maps = 200
        max_uanum = 1
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        print(f"Testing JSAP-IAP planner with use_IAP={use_AVP} and num_iterations={args.num_iterations} and max_depth={args.max_depth}")
    
    elif args.planner == 'jsapdap':
        use_AVP=False
        use_DAP = True
        args.num_drones = 1
        assert args.num_ugvs > 0
        args.max_depth = 15
        # max_uanum = max_uanum
        args.num_iterations = 1000
        args.n_maps = 200
        max_uanum = 1
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        print(f"Testing JSAP-DAP planner with use_DAP={use_DAP} and num_iterations={args.num_iterations} and max_depth={args.max_depth}")
    else:
        raise ValueError(f'Planner {args.planner} not recognized')
    
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(args.num_ugvs)]
    planner_robots = [robot.copy() for robot in robots]
    planner_drones = [drone.copy() for drone in drones]
    
    jsapplanner = planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, \
                uavs=planner_drones, rollout_fn=jsap.decsctp_rollout, C=args.C, revisit_pen=param.REVISIT_PEN, \
                rollout_num=args.num_iterations, tree_depth=args.max_depth, n_maps=args.n_maps, use_DAP=use_DAP, \
                use_AVP=use_AVP, max_uanum=max_uanum, verbose=True)
    
    
    plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=jsapplanner.reached_goal, verbose=True)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    print(f"{args.planner}: working on a team of {len(robots)} UGVs - {len(drones)} UAVs, seed {args.seed}, iter. {args.num_iterations}")
    
    for step_data in plan_exec:
        jsapplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_action, cost = jsapplanner.compute_joint_action()
        # break
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        plan_exec.save_joint_actions(joint_action, cost)
        
    print(f"The sampling time: {jsap.JSAPState.total_sampling_time}s")
    robots_net_times = [robot.net_time for robot in robots]
    
    cost_sum = np.sum(robots_net_times)
    cost_aver = np.average(robots_net_times)

    runtime = time.perf_counter() - start_time
    # average_step_time /= count_steps
    print(f"The total time: {runtime}") 
    average_step_time = 0.0    
    gpaths = []
    for robot in robots:
        x_g = [pose[0] for pose in robot.all_poses]
        y_g = [pose[1] for pose in robot.all_poses]
        gpaths.append([x_g, y_g])
    
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    goals_cords = [goal.coord for goal in goals]
    starts_cords = [start.coord for start in starts]
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpaths=gpaths, dpaths=dpaths, \
                    graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
                        seed=args.seed, cost=cost_sum, ttime=runtime, stime=average_step_time, verbose=verbose)
    plt.savefig(f'{args.save_dir}/figures/sctp_eval_planner_{args.planner}_seed_{args.seed}_{args.num_drones}UAVs.pdf')    
    plt.show()


