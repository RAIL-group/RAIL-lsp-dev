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
    # args.seed = 3009
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    
    parser.add_argument('--save_dir', type=str, default='data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='base')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_ugvs', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--C', type=int, default=300)
    parser.add_argument('--max_depth', type=int, default=500)
    parser.add_argument('--n_maps', type=int, default=80)
    parser.add_argument('--n_vertex', type=int, default=14)

    args = parser.parse_args(['--save_dir', ''])
    
    args.save_dir = 'data/sctp'
    args.planner = 'ctp'
    # args.num_drones = 
    args.num_iterations = 5000
    args.C = 200
    args.max_depth = 30
    args.current_seed = args.seed
    args.num_ugvs = 2
    
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
    # args.planner = 'jsap'
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.num_iterations = 1000

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
    

def test_jsap_plan_exec_mgraph():
    print()
    args = _get_args()
    args.planner = 'ctp'
    args.num_iterations = 1000
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.num_drones = 60
    

    starts, goals, graph = graphs.m_graph_unc()
    num_uav = 1
    num_ugv = 3
    
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    use_AVP = True 
    max_uanum = 1
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(num_ugv)]
    if args.planner == 'ctp':
        drones = []
        args.num_iterations = 1500
        args.max_depth = 15
    else:
        args.max_depth = 8
        if use_AVP:
            max_uanum = num_uav
            args.num_iterations = 500
            args.planner = 'jsap-avp'
        else:
            max_uanum = 1
            args.num_iterations = 2000
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
    args.planner = 'jsap2'
    args.seed = 3001
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.num_ugvs = 1
    starts, goals, graph = graphs.random_graph(n_vertex=args.n_vertex, SG_pairs=args.num_ugvs)
    # print(f"The iniital graph")
    # graph.print_graph_config()
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    
    
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(args.num_ugvs)]
    if args.planner == 'ctp':
        drones = []
        args.num_iterations = 2000 + 500*args.num_ugvs
        args.max_depth = 20
        use_AVP = False
    elif args.planner == 'jsap':
        use_AVP = False
        args.max_depth = 15
        assert args.num_drones > 0
        assert args.num_ugvs > 0
        max_uanum = 1
        args.num_iterations = 3500*(args.num_drones)
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
    elif args.planner == 'jsap2':
        use_AVP = False
        args.num_drones = 2
        args.max_depth = 15
        param.REVISIT_PEN = 0.0
        assert args.num_drones == 2
        assert args.num_ugvs == 1 
        max_uanum = 1
        args.num_iterations = 6000+3500*(args.num_drones-1)
        drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, \
                robot_type=RobotType.Drone, at_node=True) for _ in range(args.num_drones)]

    elif args.planner == 'jsapavp':
        use_AVP==True
        assert args.num_drones > 0
        assert args.num_ugvs > 0
        args.max_depth = 8
        max_uanum = max_uanum
        args.num_iterations = 500
        max_uanum = 1
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        print(f"Testing JSAP-AVP planner with use_AVP={use_AVP} and num_iterations={args.num_iterations} and max_depth={args.max_depth}")
    else:
        raise ValueError(f'Planner {args.planner} not recognized')
    
    planner_robots = [robot.copy() for robot in robots]
    planner_drones = [drone.copy() for drone in drones]
    jsapplanner = planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, \
                uavs=planner_drones, rollout_fn=jsap.decsctp_rollout, C=args.C, revisit_pen=param.REVISIT_PEN, \
                rollout_num=args.num_iterations, tree_depth=args.max_depth, n_maps=args.n_maps, \
                use_AVP=use_AVP, max_uanum=max_uanum, verbose=True)
    plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=jsapplanner.reached_goal, verbose=True)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    print(f"{args.planner}: working on a team of {len(robots)} UGVs and {len(drones)} UAVs with seed {args.seed}")
    
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
    
    robots_net_times = [robot.net_time for robot in robots]
    
    cost_sum = np.sum(robots_net_times)
    cost_aver = np.average(robots_net_times)

    runtime = time.perf_counter() - start_time
    average_step_time /= count_steps
    # average_step_time = 0.0    
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
                        seed=args.seed, cost=cost_sum, ttime=runtime, stime=average_step_time, verbose=True)
    plt.show()


# def test_decPrior_plan_exec_bridges_graph():
#     print()
#     args = _get_args()
#     args.seed = 3016
#     args.planner = 'dsap_avp'
#     args.num_iterations = 1000
#     args.n_maps = 80
#     args.max_depth = 20
#     random.seed(args.seed)
#     np.random.seed(args.seed)

#     start, goal, graph = graphs.random_bridges_graph()
    
#     num_uav = 1
#     num_ugv = 1
#     starts = [start]
#     goals = [goal]
#     plotGraph = graph.copy()
#     policyGraph = graph.copy()
#     use_AVP = True 
#     max_uanum = 1
#     robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
#                     at_node=True) for i in range(num_ugv)]
#     drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
#                 robot_type=RobotType.Drone, at_node=True) for i in range(num_uav)]
    
#     robots_copy = [robot.copy() for robot in robots]
#     drones_copy = [drone.copy() for drone in drones]
    
#     assert args.n_maps == 80
#     decPriorplanner = planner.DecPriorPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=robots_copy,
#                                             uavs=drones_copy, rollout_fn=dec_prior.decsctp_rollout, C=args.C, 
#                                             rollout_num=args.num_iterations, tree_depth=args.max_depth, n_maps=args.n_maps, 
#                                             use_2AG=use_AVP, max_uanum=max_uanum, verbose=True)
#     plan_exec = plan_loop.DecPriorPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
#                                                     reached_goal=decPriorplanner.reached_goal, verbose=True)

#     start_time = time.perf_counter() 
#     average_step_time = 0.0
#     count_steps = 0
    
#     for step_data in plan_exec:
#         decPriorplanner.update(
#             step_data['observed_pois'],
#             step_data['ugvs'],
#             step_data['uavs']
#         )
#         assert policyGraph.pois[1].block_status == graph.pois[1].block_status
#         assert policyGraph.pois[1].block_prob == graph.pois[1].block_prob
#         time1 = time.perf_counter()
#         joint_action, cost = decPriorplanner.compute_joint_action()
#         average_step_time += (time.perf_counter() - time1)
#         count_steps += 1
#         plan_exec.save_joint_actions(joint_action, cost)
    
#     cost_sum = np.sum([robot.net_time for robot in robots])
#     cost_aver = np.average([robot.net_time for robot in robots])
#     # print(f"Cost sum: {cost_sum}, Cost average: {cost_aver}, and true value: {robots[0].net_time}")

#     runtime = time.perf_counter() - start_time
#     average_step_time /= count_steps    
#     gpaths = []
#     for robot in robots:
#         x_g = [pose[0] for pose in robot.all_poses]
#         y_g = [pose[1] for pose in robot.all_poses]
#         gpaths.append([x_g, y_g])
    
#     dpaths = []
#     for drone in drones:
#         x = [pose[0] for pose in drone.all_poses]
#         y = [pose[1] for pose in drone.all_poses]
#         dpaths.append([x, y])
#     goals_cords = [goal.coord for goal in goals]
#     starts_cords = [start.coord for start in starts]
#     plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpaths=gpaths, dpaths=dpaths, \
#                     graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
#                         seed=args.seed, cost=cost_aver, verbose=False)
    
#     # if need_pdf:
#     plt.savefig(f'../data/sctp/tests/sctp_test_planner_{args.planner}_seed_{args.seed}.pdf')

#     logfile = f'../data/sctp/tests/results_{args.num_ugvs}UGV.txt'
#     with open(logfile, "a+") as f:
#         f.write(f"SEED: {args.seed} | UAVs: {args.num_drones} | PLANNER: {args.planner} | SUCC: {int(plan_exec.success)} "
#                 f"| COST_AVER: {cost_aver:0.3f} | COST_SUM: {cost_sum:0.3f} | T.TIME: {runtime:0.2f} | STEP.TIME : {average_step_time:0.2f} "
#                 f"| SAMP.TIME : {decPriorplanner.sampling_time:0.2f} | SPOLICY.TIME : {decPriorplanner.single_policy_time:0.2f}\n")    

#     # plt.show()

# def test_decPrior_plan_exec_sgraph_2goals():
#     print()
    
#     args = _get_args()
#     args.planner = 'sctp'
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     args.num_iterations = 800

#     starts, goals, graph = graphs.s_graph_2goals()
#     num_uav = 1
#     num_ugv = 1
#     for poi in graph.pois:
#         if poi.id == 9 or poi.id==6 or poi.id==10:
#             poi.block_status = 0
#         elif poi.id == 11:
#             poi.block_status = 1
    
#     plotGraph = graph.copy()
#     use_2AG = True 
#     max_uanum = 1
#     robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
#                     at_node=True) for i in range(num_ugv)]
#     drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
#                 robot_type=RobotType.Drone, at_node=True) for i in range(num_uav)]
#     decPriorplanner = planner.DecPriorPlanner(init_graph=graph, goalIDs=[goal.id for goal in goals], ugvs=robots, uavs=drones,
#                                             rollout_fn=dec_prior.decsctp_rollout, C=args.C, rollout_num=args.num_iterations,
#                                             tree_depth=args.max_depth, n_maps=args.n_maps, use_2AG=use_2AG,
#                                             max_uanum=max_uanum, verbose=True)
#     plan_exec = plan_loop.DecPriorPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
#                                                     reached_goal=decPriorplanner.reached_goal)

#     start_time = time.perf_counter() 
#     average_step_time = 0.0
#     count_steps = 0
    
#     for step_data in plan_exec:
#         decPriorplanner.update(
#             step_data['observed_pois'],
#             step_data['ugvs'],
#             step_data['uavs']
#         )
#         time1 = time.perf_counter()
#         # if (count_steps==1):
#         #     print("Done to the first loop:")
            
#         joint_action, cost = decPriorplanner.compute_joint_action()
#         average_step_time += (time.perf_counter() - time1)
#         count_steps += 1
#         plan_exec.save_joint_actions(joint_action, cost)
    
#     cost_sum = np.sum([robot.net_time for robot in robots])
#     cost_aver = np.average([robot.net_time for robot in robots])

#     runtime = time.perf_counter() - start_time
#     average_step_time /= count_steps    
#     gpaths = []
#     for robot in robots:
#         x_g = [pose[0] for pose in robot.all_poses]
#         y_g = [pose[1] for pose in robot.all_poses]
#         gpaths.append([x_g, y_g])
    
#     dpaths = []
#     for drone in drones:
#         x = [pose[0] for pose in drone.all_poses]
#         y = [pose[1] for pose in drone.all_poses]
#         dpaths.append([x, y])
#     goals_cords = [goal.coord for goal in goals]
#     starts_cords = [start.coord for start in starts]
#     plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpaths=gpaths, dpaths=dpaths, \
#                     graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
#                         seed=args.seed, cost=cost_aver, verbose=False)
#     plt.savefig(f'../data/sctp/tests/sctp_test_planner_{args.planner}_seed_{args.seed}.pdf')
#     plt.show()

