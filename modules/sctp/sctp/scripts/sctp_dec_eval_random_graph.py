import numpy as np
import argparse, time, random
from sctp.utils import plotting
# import random, time
from sctp import sctp_graphs as graphs
from sctp.robot import Robot
from sctp import core, param
import matplotlib.pyplot as plt
from sctp.param import RobotType
from pathlib import Path
from sctp.planners.sctp_planner_dec import GroundPlanner
from sctp.planners.sctp_planner_dec import DronesPlanner
from sctp.planners.sctp_exe_dec import SCTPDecExe
from sctp import dstate_dec


def _setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.num_drones ==0:
        args.planner = 'ctp'
    # print(f"Random graph -number of rollouts: {args.num_iterations}")
    starts, goals, graph = graphs.random_graph(n_vertex=args.n_vertex, SG_pairs=args.num_grounds)
    plotGraph = graph.copy()
    graphs_copy = [graph.copy() for _ in range(args.num_grounds+1)]
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, at_node=True) 
                                for i in range(args.num_grounds)]
    for i, robot in enumerate(robots):
        robot.id = i+1
    drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, \
                    robot_type=RobotType.Drone, at_node=True) \
                    for i in range(args.num_drones)]
    for i, drone in enumerate(drones):
        drone.id = i+1
        drone.unfinished_action = None
        
    ground_planners =  [GroundPlanner(init_graph=graphs_copy[i], goalID=goals[i].id, robot=robots[i].copy(), \
                                    rollout_fn=core.sctp_rollout3, num_rollouts=int(args.num_iterations),
                                    C=args.C, tree_depth=args.max_depth, sampling_maps=args.sampling_maps, verbose=False) \
                                    for i in range(args.num_grounds)]
    if len(drones) > 0: 
        drones_copy = [drone.copy() for drone in drones]
        drones_planner = DronesPlanner(init_graph=graphs_copy[-1], goalID=goals[0].id, drones=drones_copy, \
                                        rollout_fn=dstate_dec.drone_rollout, num_rollouts=int(3*args.num_iterations),
                                        C=args.C, sampling_maps=args.sampling_maps, tree_depth=int(2*args.max_depth), verbose=False)
    
    
    plan_exec = SCTPDecExe(graph=graph, robots=robots, drones=drones, goalIDs=[g.id for g in goals], verbose=False)
    count_planning_drone = 0
    count_planning_ground = 0
    start_time = time.perf_counter()
    aver_runtime_ground = 0.0
    aver_runtime_drones = 0.0
    for step_data in plan_exec:
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
        
        drones_subtargets = []
        robots_actions = []
        robots_costs = [] 
        drones_actions = []
        drones_costs = []
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
            aver_runtime_ground += time.perf_counter()-time1
            count_planning_ground +=1
            time1 = time.perf_counter()
            
        if len(drones) > 0:
            drones_subactions = []
            time1 = time.perf_counter()
            for target in drones_subtargets:
                drones_subactions.append(core.Action(target=target, rtype=RobotType.Drone))
            drones_actions, drones_costs = drones_planner.compute_joint_action(sub_actions=drones_subactions)
            aver_runtime_drones += time.perf_counter()-time1
            count_planning_drone +=1
        plan_exec.save_joint_actions(robots_actions, robots_costs, drones_actions, drones_costs)
    cost = max([robot.net_time for robot in robots])
    aver_runtime_ground /= float(count_planning_ground)
    
    if len(drones) > 0:
        aver_runtime_drones /= float(count_planning_drone)
    else:
        aver_runtime_drones = 0.0
    
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
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')

    logfile = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | SUCC : {int(plan_exec.success)} | COST : {cost:0.3f} | T.TIME : {runtime:0.2f} | D.TIME :  {aver_runtime_drones:0.2f} | G.TIME :  {aver_runtime_ground:0.2f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='base')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_grounds', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=750)
    parser.add_argument('--C', type=float, default=300.0)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--sampling_maps', type=int, default=80)
    parser.add_argument('--n_vertex', type=int, default=18)
 
    args = parser.parse_args()
    args.current_seed = args.seed
    print("Reaching here")
    _setup(args)
