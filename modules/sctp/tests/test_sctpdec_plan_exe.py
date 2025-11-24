import random, time, pytest, argparse
import numpy as np
import matplotlib.pyplot as plt
# import argparse
from pathlib import Path
from sctp import sctp_graphs as graphs
from sctp.utils import plotting 
from sctp.robot import Robot
from sctp import core, param
from sctp.param import EventOutcome, RobotType
from sctp.planners.sctp_planner_dec import GroundPlanner, DronesPlanner
from sctp.planners.sctp_exe_dec import SCTPDecExe
from sctp import dstate_dec


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--planner', type=str, default='base')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_ground', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--C', type=int, default=300)
    parser.add_argument('--max_depth', type=int, default=100)
    parser.add_argument('--sampling_maps', type=int, default=300)
    parser.add_argument('--n_vertex', type=int, default=14)

    args = parser.parse_args(['--save_dir', ''])
    args.seed = 3000
    args.save_dir = '/data/sctp'
    args.planner = 'sctp'
    args.num_drones = 0
    args.num_ground = 1
    args.num_iterations = 1200
    args.C = 200
    args.max_depth = 35
    args.sampling_maps = 300
    args.current_seed = args.seed
    
    return args


def test_sctpdec_plan_exec_lg():
    print()
    args = _get_args()
    args.planner = 'sctp'
    random.seed(args.seed)
    np.random.seed(args.seed)

    start, goal, l_graph = graphs.linear_graph_unc()
    starts = [start, start]
    goals = [goal, goal]
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, at_node=True) 
                                for i in range(args.num_ground)]
    for i, robot in enumerate(robots):
        robot.id = i
    drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, \
                    robot_type=RobotType.Drone, at_node=True) \
                    for i in range(args.num_drones)]
    for i, drone in enumerate(drones):
        drone.id = i
    
    ground_planners =  [GroundPlanner(args=args, init_graph=l_graph, goalID=goals[i].id, robot=robots[i].copy(), \
                                    rollout_fn=core.sctp_rollout3, verbose=True) \
                                    for i in range(args.num_ground)]
    drones_copy = [drone.copy() for drone in drones]
    drones_planner = DronesPlanner(args=args, init_graph=l_graph, goalID=goals[0].id, drones=drones_copy, \
                                    rollout_fn=dstate_dec.drone_rollout, tree_depth=50, sampling_maps=100, verbose=True)
    
    plan_exec = SCTPDecExe(graph=l_graph, robots=robots, drones=drones, goalIDs=[g.id for g in goals])
    count = 0
    for step_data in plan_exec:
        print("####################### New navigation #######################################")
        drones_planner.update(
            step_data['observed_pois'],
            step_data['drones'],
            step_data['robots']
            
        )
        for i, ground_planner in enumerate(ground_planners):
            ground_planner.update(
                step_data['observed_pois'],
                step_data['robots'][i]
            )
        
        #### Compute the policy - next action
        drones_subactions = []
        robots_actions = []
        robots_costs = [] 
        print("----------------- planning for ground -------------------")
        for i, ground_planner in enumerate(ground_planners):
            if ground_planner.reached_goal():
                continue
            g_ordering, g_costs = ground_planner.compute_action()
            robots_actions.append(g_ordering)
            robots_costs.append(g_costs)
            for action in g_ordering:
                drones_subactions.append(action)
        print("----------------- planning for drones -------------------")
        drones_actions, drones_costs = drones_planner.compute_joint_action(sub_actions=drones_subactions)
        plan_exec.save_joint_actions(robots_actions, robots_costs, drones_actions, drones_costs)
        if count > 5:
            break # only one step for testing
        count += 1
    
def test_sctpdec_plan_exec_dgraph():
    print()
    args = _get_args()
    args.planner = 'sctp'
    random.seed(args.seed)
    np.random.seed(args.seed)

    start, goal, graph = graphs.disjoint_unc()
    plotGraph = graph.copy()
    starts = [start, start]
    goals = [goal, goal]
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, at_node=True) 
                                for i in range(args.num_ground)]
    for i, robot in enumerate(robots):
        robot.id = i
    drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, \
                    robot_type=RobotType.Drone, at_node=True) \
                    for i in range(args.num_drones)]
    for i, drone in enumerate(drones):
        drone.id = i
        if i == 0:
            drone.unfinished_action = core.Action(target=7, rtype=RobotType.Drone, start_pose=drone.cur_pose)
    
    ground_planners =  [GroundPlanner(args=args, init_graph=graph, goalID=goals[i].id, robot=robots[i].copy(), \
                                    rollout_fn=core.sctp_rollout3, verbose=False) \
                                    for i in range(args.num_ground)]
    drones_copy = [drone.copy() for drone in drones]
    drones_planner = DronesPlanner(args=args, init_graph=graph, goalID=goals[0].id, drones=drones_copy, \
                                    rollout_fn=dstate_dec.drone_rollout, tree_depth=10, sampling_maps=100, verbose=False)
    
    
    plan_exec = SCTPDecExe(graph=graph, robots=robots, drones=drones, goalIDs=[g.id for g in goals], verbose=True)
    count = 0
    start_time = time.perf_counter() 
    for step_data in plan_exec:
        print("####################### New navigation #######################################")
        drones_planner.update(
            step_data['observed_pois'],
            step_data['drones'],
            step_data['robots']
            
        )
        for i, ground_planner in enumerate(ground_planners):
            ground_planner.update(
                step_data['observed_pois'],
                step_data['robots'][i]
            )
        
        #### Compute the policy - next action
        drones_subtargets = []
        robots_actions = []
        robots_costs = [] 
        # print("----------------- planning for ground -------------------")
        for i, ground_planner in enumerate(ground_planners):
            if ground_planner.reached_goal():
                continue
            g_ordering, g_costs = ground_planner.compute_action()
            robots_actions.append(g_ordering)
            robots_costs.append(g_costs)
            for action in g_ordering:
                drones_subtargets.append(action.target)
        # print("----------------- planning for drones -------------------")
        drones_subactions = []
        for target in drones_subtargets:
            drones_subactions.append(core.Action(target=target, rtype=RobotType.Drone))
        drones_actions, drones_costs = drones_planner.compute_joint_action(sub_actions=drones_subactions)
        plan_exec.save_joint_actions(robots_actions, robots_costs, drones_actions, drones_costs)
        # if count > 5:
            # break # only one step for testing
        # count += 1
    
    cost = max([robot.net_time for robot in robots])
    print(f"The cost (time) to reach the goal is: {cost}")

    runtime = time.perf_counter() - start_time
    x_g = [pose[0] for pose in robots[0].all_poses]
    y_g = [pose[1] for pose in robots[0].all_poses]
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drones[0].all_poses]
        y = [pose[1] for pose in drones[0].all_poses]
        dpaths.append([x, y])
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpath=[x_g, y_g], dpaths=dpaths, graph_plot=plotGraph,
                             start_coord=start.coord, goal_coord=goal.coord, seed=args.seed, cost=cost, verbose=True)
    # plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')
    plt.show()

def test_sctpdec_plan_exec_sgraph():
    print()
    args = _get_args()
    args.planner = 'sctp'
    args.num_drones = 2
    random.seed(args.seed)
    np.random.seed(args.seed)

    start, goal, graph = graphs.s_graph_unc()
    plotGraph = graph.copy()
    starts = [start, start]
    goals = [goal, goal]
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, at_node=True) 
                                for i in range(args.num_ground)]
    for i, robot in enumerate(robots):
        robot.id = i
    drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, \
                    robot_type=RobotType.Drone, at_node=True) \
                    for i in range(args.num_drones)]
    for i, drone in enumerate(drones):
        drone.id = i
        if i == 0:
            drone.unfinished_action = core.Action(target=7, rtype=RobotType.Drone, start_pose=drone.cur_pose)
    
    ground_planners =  [GroundPlanner(args=args, init_graph=graph, goalID=goals[i].id, robot=robots[i].copy(), \
                                    rollout_fn=core.sctp_rollout3, verbose=False) \
                                    for i in range(args.num_ground)]
    drones_copy = [drone.copy() for drone in drones]
    drones_planner = DronesPlanner(args=args, init_graph=graph, goalID=goals[0].id, drones=drones_copy, \
                                    rollout_fn=dstate_dec.drone_rollout, tree_depth=10, sampling_maps=100, verbose=False)
    
    plan_exec = SCTPDecExe(graph=graph, robots=robots, drones=drones, goalIDs=[g.id for g in goals], verbose=True)
    count = 0
    start_time = time.perf_counter() 
    for step_data in plan_exec:
        print("####################### New navigation #######################################")
        drones_planner.update(
            step_data['observed_pois'],
            step_data['drones'],
            step_data['robots']
            
        )
        for i, ground_planner in enumerate(ground_planners):
            ground_planner.update(
                step_data['observed_pois'],
                step_data['robots'][i]
            )
        
        drones_subtargets = []
        robots_actions = []
        robots_costs = [] 
        for i, ground_planner in enumerate(ground_planners):
            if ground_planner.reached_goal():
                continue
            g_ordering, g_costs = ground_planner.compute_action()
            robots_actions.append(g_ordering)
            robots_costs.append(g_costs)
            for action in g_ordering:
                drones_subtargets.append(action.target)
        drones_subactions = []
        for target in drones_subtargets:
            drones_subactions.append(core.Action(target=target, rtype=RobotType.Drone))
        drones_actions, drones_costs = drones_planner.compute_joint_action(sub_actions=drones_subactions)
        plan_exec.save_joint_actions(robots_actions, robots_costs, drones_actions, drones_costs)
    
    cost = max([robot.net_time for robot in robots])
    print(f"The cost (time) to reach the goal is: {cost}")

    runtime = time.perf_counter() - start_time
    gpaths = []
    for robot in robots:
        # gpaths.append([pose for pose in robot.all_poses])
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
                        seed=args.seed, cost=cost, verbose=True)
    # plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')
    plt.show()


def test_sctpdec_plan_exec_mgraph():
    print()
    args = _get_args()
    args.planner = 'sctp'
    args.num_drones = 1
    args.num_ground = 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.num_iterations =750
    args.sampling_maps =80
    args.max_depth =7

    starts, goals, graph = graphs.m_graph_unc()
    # print("Get the avaiable graph")
    plotGraph = graph.copy()
    graphs_copy = [graph.copy() for _ in range(args.num_ground+1)]
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, at_node=True) 
                                for i in range(args.num_ground)]
    for i, robot in enumerate(robots):
        robot.id = i
    drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, \
                    robot_type=RobotType.Drone, at_node=True) \
                    for i in range(args.num_drones)]
    for i, drone in enumerate(drones):
        drone.id = i
        if i == 0:
            drone.unfinished_action = core.Action(target=15, rtype=RobotType.Drone, start_pose=drone.cur_pose)
    
    ground_planners =  [GroundPlanner(init_graph=graphs_copy[i], goalID=goals[i].id, robot=robots[i].copy(), \
                                    rollout_fn=core.sctp_rollout3, num_rollouts=int(args.num_iterations),
                                    C=args.C, tree_depth=args.max_depth, sampling_maps=args.sampling_maps, verbose=True) \
                                    for i in range(args.num_ground)]
    drones_copy = [drone.copy() for drone in drones]
    drones_planner = DronesPlanner(init_graph=graphs_copy[-1], goalID=goals[0].id, drones=drones_copy, \
                                    rollout_fn=dstate_dec.drone_rollout, num_rollouts=int(3*args.num_iterations),
                                    C=args.C, sampling_maps=args.sampling_maps, tree_depth=int(2*args.max_depth), verbose=True)
    
    
    plan_exec = SCTPDecExe(graph=graph, robots=robots, drones=drones, goalIDs=[g.id for g in goals], verbose=True)
    count = 0
    start_time = time.perf_counter() 
    for step_data in plan_exec:
        print("####################### New navigation #######################################")
        drones_planner.update(
            step_data['observed_pois'],
            step_data['drones'],
            step_data['robots']
            
        )
        for i, ground_planner in enumerate(ground_planners):
            ground_planner.update(
                step_data['observed_pois'],
                step_data['robots'][i]
            )
        
        #### Compute the policy - next action
        drones_subtargets = []
        robots_actions = []
        robots_costs = [] 
        # print("----------------- planning for ground -------------------")
        time1 = time.perf_counter()
        for i, ground_planner in enumerate(ground_planners):
            if ground_planner.reached_goal():
                robots_actions.append([core.Action(target=goals[i].id, rtype=RobotType.Ground, start_pose=robots[i].cur_pose)])
                robots_costs.append([0.0])
                continue
            g_ordering, g_costs = ground_planner.compute_action()
            robots_actions.append(g_ordering)
            robots_costs.append(g_costs)
            for action in g_ordering:
                drones_subtargets.append(action.target)
            print(f"Planning time for robot {i} is {time.perf_counter()-time1:0.3f} seconds")
            time1 = time.perf_counter()
        # print("----------------- planning for drones -------------------")
        drones_subactions = []
        time1 = time.perf_counter()
        for target in drones_subtargets:
            drones_subactions.append(core.Action(target=target, rtype=RobotType.Drone))
        print(f"The number of drone subactions is {len(drones_subactions)}")
        drones_actions, drones_costs = drones_planner.compute_joint_action(sub_actions=drones_subactions)
        plan_exec.save_joint_actions(robots_actions, robots_costs, drones_actions, drones_costs)
        print(f"Planning time for drones is {time.perf_counter()-time1:0.3f} seconds")
    
    cost = max([robot.net_time for robot in robots])
    print(f"The cost (time) to reach the goal is: {cost}")

    runtime = time.perf_counter() - start_time
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
                        seed=args.seed, cost=cost, verbose=True)
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}_Oct17.png')
    plt.show()

def test_sctpdec_plan_exec_rangraph():
    print()
    args = _get_args()
    args.planner = 'sctp'
    args.num_drones = 0
    args.num_ground = 1
    args.seed = 3000
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.num_iterations =800
    args.sampling_maps =80
    args.max_depth =10
    args.n_vertex = 18
    print("Generating random graph")
    starts, goals, graph = graphs.random_graph(n_vertex=args.n_vertex, SG_pairs=args.num_ground)
    print("Get the avaiable graph")
    plotGraph = graph.copy()
    graphs_copy = [graph.copy() for _ in range(args.num_ground+1)]
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, at_node=True) 
                                for i in range(args.num_ground)]
    for i, robot in enumerate(robots):
        robot.id = i
    drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, \
                    robot_type=RobotType.Drone, at_node=True) \
                    for i in range(args.num_drones)]
    for i, drone in enumerate(drones):
        drone.unfinished_action = None
        
    ground_planners =  [GroundPlanner(init_graph=graphs_copy[i], goalID=goals[i].id, robot=robots[i].copy(), \
                                    rollout_fn=core.sctp_rollout3, num_rollouts=int(args.num_iterations),
                                    C=args.C, tree_depth=args.max_depth, sampling_maps=args.sampling_maps, verbose=True) \
                                    for i in range(args.num_ground)]
    drones_copy = [drone.copy() for drone in drones]
    drones_planner = DronesPlanner(init_graph=graphs_copy[-1], goalID=goals[0].id, drones=drones_copy, \
                                    rollout_fn=dstate_dec.drone_rollout, num_rollouts=int(3*args.num_iterations),
                                    C=args.C, sampling_maps=args.sampling_maps, tree_depth=int(2*args.max_depth), verbose=True)
    
    
    plan_exec = SCTPDecExe(graph=graph, robots=robots, drones=drones, goalIDs=[g.id for g in goals], verbose=True)
    count = 0
    start_time = time.perf_counter() 
    for step_data in plan_exec:
        print("####################### New navigation #######################################")
        if len(drones) > 0:
            drones_planner.update(
                step_data['observed_pois'],
                step_data['drones'],
                step_data['robots']
            )
        for i, ground_planner in enumerate(ground_planners):
            ground_planner.update(
                step_data['observed_pois'],
                step_data['robots'][i]
            )
        
        #### Compute the policy - next action
        drones_subtargets = []
        robots_actions = []
        drones_actions = []
        drones_costs = []
        robots_costs = [] 
        # print("----------------- planning for ground -------------------")
        time1 = time.perf_counter()
        for i, ground_planner in enumerate(ground_planners):
            if ground_planner.reached_goal():
                robots_actions.append([core.Action(target=goals[i].id, rtype=RobotType.Ground, start_pose=robots[i].cur_pose)])
                robots_costs.append([0.0])
                continue
            g_ordering, g_costs = ground_planner.compute_action()
            robots_actions.append(g_ordering)
            robots_costs.append(g_costs)
            for action in g_ordering:
                drones_subtargets.append(action.target)
            print(f"Planning time for robot {i} is {time.perf_counter()-time1:0.3f} seconds")
            time1 = time.perf_counter()
        # print("----------------- planning for drones -------------------")
        if len(drones) > 0:
            drones_subactions = []
            time1 = time.perf_counter()
            for target in drones_subtargets:
                drones_subactions.append(core.Action(target=target, rtype=RobotType.Drone))
            print(f"The number of drone subactions is {len(drones_subactions)}")
            drones_actions, drones_costs = drones_planner.compute_joint_action(sub_actions=drones_subactions)
        plan_exec.save_joint_actions(robots_actions, robots_costs, drones_actions, drones_costs)
        print(f"Planning time for drones is {time.perf_counter()-time1:0.3f} seconds")
    
    cost = max([robot.net_time for robot in robots])
    print(f"The cost (time) to reach the goal is: {cost}")

    runtime = time.perf_counter() - start_time
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
                        seed=args.seed, cost=cost, verbose=True)
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}_Oct17.png')
    plt.show()


# if __name__ == '__main__':
#     # test_sctp_planner_lg()
#     test_sctp_planner_sg()    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--save_dir', type=str, default='/data/sctp')
    # parser.add_argument('--seed', type=int, default=1024)
    # parser.add_argument('--planner', type=str, default='base')
    # parser.add_argument('--num_drones', type=int, default=1)
    # parser.add_argument('--num_iterations', type=int, default=20)
    # parser.add_argument('--C', type=float, default=200)
    # args = parser.parse_args()
    # args.current_seed = args.seed
    
    
    # # test_dronesplanner_sg(args=args)
    # test_sctpdec_plan_exec_lg()

