import random, time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sctp import sctp_graphs as graphs
from sctp.utils import plotting 
from sctp.robot import Robot
from sctp import core, param, dec_prior
from sctp.param import EventOutcome, RobotType
from sctp.planners import decPrior_planner as planner
from sctp.planners import decPrior_plan_exe as plan_loop


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='base')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--C', type=int, default=300)
    parser.add_argument('--max_depth', type=int, default=500)
    parser.add_argument('--n_maps', type=int, default=100)
    parser.add_argument('--n_vertex', type=int, default=14)

    args = parser.parse_args(['--save_dir', ''])
    args.seed = 2007
    args.save_dir = '/data/sctp'
    args.planner = 'sctp'
    args.num_drones = 1
    args.num_iterations = 1000
    args.C = 30
    args.max_depth = 35
    args.current_seed = args.seed
    
    return args


def test_decPrior_plan_exec_lg():
    print()
    args = _get_args()
    args.planner = 'sctp'
    random.seed(args.seed)
    np.random.seed(args.seed)

    starts, goals, graph = graphs.linear_graph_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        poi.block_status = 0
    
    plotGraph = graph.copy()
    use_2AG = True 
    max_uanum = 1
    
    
    robots = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, at_node=True)]
    drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, robot_type=RobotType.Drone, at_node=True)]
    decPriorplanner = planner.DecPriorPlanner(init_graph=graph, goalIDs=[goal.id for goal in goals], ugvs=robots, uavs=drones,
                                            rollout_fn=dec_prior.decsctp_rollout, C=args.C, rollout_num=args.num_iterations,
                                            tree_depth=args.max_depth, n_maps=args.n_maps, use_2AG=use_2AG,
                                            max_uanum=max_uanum, verbose=True)
    plan_exec = plan_loop.DecPriorPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=decPriorplanner.reached_goal)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    
    for step_data in plan_exec:
        decPriorplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_action, cost = decPriorplanner.compute_joint_action()
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        # plan_exec.update_joint_action(joint_action, cost)
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
                        seed=args.seed, cost=cost_aver, verbose=True)
    plt.show()
    

def test_decPrior_plan_exec_dg():
    print()
    args = _get_args()
    args.planner = 'sctp'
    random.seed(args.seed)
    np.random.seed(args.seed)

    starts, goals, graph = graphs.disjoint_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        if poi.id == 8 or poi.id==6 or poi.id==5:
            poi.block_status = 0
        else:
            poi.block_status = 1
    
    plotGraph = graph.copy()
    use_2AG = True 
    max_uanum = 1
    robots = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, at_node=True)]
    drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, robot_type=RobotType.Drone, at_node=True)]
    decPriorplanner = planner.DecPriorPlanner(init_graph=graph, goalIDs=[goal.id for goal in goals], ugvs=robots, uavs=drones,
                                            rollout_fn=dec_prior.decsctp_rollout, C=args.C, rollout_num=args.num_iterations,
                                            tree_depth=args.max_depth, n_maps=args.n_maps, use_2AG=use_2AG,
                                            max_uanum=max_uanum, verbose=True)
    plan_exec = plan_loop.DecPriorPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=decPriorplanner.reached_goal)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    
    for step_data in plan_exec:
        decPriorplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_action, cost = decPriorplanner.compute_joint_action()
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        # plan_exec.update_joint_action(joint_action, cost)
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
                        seed=args.seed, cost=cost_aver, verbose=False)
    plt.show()
    

def test_decPrior_plan_exec_sgraph_2goals():
    print()
    args = _get_args()
    args.planner = 'sctp'
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.num_iterations = 800

    starts, goals, graph = graphs.s_graph_2goals()
    num_uav = 2
    num_ugv = 2
    for poi in graph.pois:
        if poi.id == 9 or poi.id==6 or poi.id==10:
            poi.block_status = 0
        elif poi.id == 11:
            poi.block_status = 1
    
    plotGraph = graph.copy()
    use_2AG = True 
    max_uanum = 3
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(num_ugv)]
    drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(num_uav)]
    decPriorplanner = planner.DecPriorPlanner(init_graph=graph, goalIDs=[goal.id for goal in goals], ugvs=robots, uavs=drones,
                                            rollout_fn=dec_prior.decsctp_rollout, C=args.C, rollout_num=args.num_iterations,
                                            tree_depth=args.max_depth, n_maps=args.n_maps, use_2AG=use_2AG,
                                            max_uanum=max_uanum, verbose=True)
    plan_exec = plan_loop.DecPriorPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=decPriorplanner.reached_goal)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    
    for step_data in plan_exec:
        decPriorplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        # if (count_steps==1):
        #     print("Done to the first loop:")
            
        joint_action, cost = decPriorplanner.compute_joint_action()
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
                        seed=args.seed, cost=cost_aver, verbose=False)
    plt.show()


def test_decPrior_plan_exec_mgraph():
    print()
    args = _get_args()
    args.planner = 'sctp'
    args.num_iterations = 3000
    random.seed(args.seed)
    np.random.seed(args.seed)

    starts, goals, graph = graphs.m_graph_unc()
    num_uav = 2
    num_ugv = 3
    
    plotGraph = graph.copy()
    use_2AG = True 
    max_uanum = 1
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(num_ugv)]
    drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                robot_type=RobotType.Drone, at_node=True) for i in range(num_uav)]
    decPriorplanner = planner.DecPriorPlanner(init_graph=graph, goalIDs=[goal.id for goal in goals], ugvs=robots, uavs=drones,
                                            rollout_fn=dec_prior.decsctp_rollout, C=args.C, rollout_num=args.num_iterations,
                                            tree_depth=args.max_depth, n_maps=args.n_maps, use_2AG=use_2AG,
                                            max_uanum=max_uanum, verbose=True)
    plan_exec = plan_loop.DecPriorPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=decPriorplanner.reached_goal)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    
    for step_data in plan_exec:
        decPriorplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_action, cost = decPriorplanner.compute_joint_action()
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        # print(f"First 4 actions: {joint_action[:4]}")
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
                        seed=args.seed, cost=cost_aver, verbose=False)
    plt.show()

