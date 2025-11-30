import numpy as np
import argparse
from sctp.utils import plotting
import random, time
from sctp import sctp_graphs as graphs
from sctp.robot import Robot
from sctp import core, param
import matplotlib.pyplot as plt
from sctp.param import RobotType
from pathlib import Path
from sctp.planners import sctp_planner as planner
from sctp.planners import sctp_plan_exe as plan_loop


def _setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    print_pdf = True
    # start, goal, graph = graphs.random_graph(n_vertex=args.n_vertex)
    starts, goals, graph = graphs.random_graph(n_vertex=args.n_vertex, SG_pairs=1)
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    robot = Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, at_node=True)
    planner_robot = robot.copy()
    # param.IV_SAMPLE_SIZE = 80
    # print(f"Running CTP with num_iterations {args.num_iterations}")
    if args.planner == 'base':
        drones = []
        args.num_drones = 0
        param.REVISIT_PEN = 20.0
    elif args.planner =='jsctp1':
        args.num_drones = 1
        drones = [Robot(position=[starts[0].coord[0], starts[0].coord[1]], cur_node=starts[0].id, robot_type=RobotType.Drone, at_node=True)]
        param.REVISIT_PEN = 0.0
        param.ADD_IV = False
    elif args.planner == 'jsctp2':
        args.num_drones = 2
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, robot_type=RobotType.Drone, at_node=True)
                    for i in range(args.num_drones)]
        param.ADD_IV = False
    elif args.planner == 'jsctpig':
        args.num_drones = 1
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, robot_type=RobotType.Drone, at_node=True)
                    for i in range(args.num_drones)]
        param.ADD_IV = True
    else:
        raise ValueError(f'Planner {args.planner} not recognized')
    
    planner_drones = [drone.copy() for drone in drones]
    sctpplanner = planner.SCTPPlanner(init_graph=policyGraph, goalID=goals[0].id,robot=planner_robot, drones=planner_drones, 
                    tree_depth=args.max_depth, C= args.C, rollout_num=args.num_iterations,
                    n_maps=args.sampling_maps, rollout_fn=core.sctp_rollout3) 
    planning_exe = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goals[0].id,\
                                                graph=graph, reached_goal=sctpplanner.reached_goal, verbose=False)

    
    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    for step_data in planning_exe:
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        time1 = time.perf_counter()
        joint_actions, cost = sctpplanner.compute_joint_action()
        # print
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        planning_exe.save_joint_actions(joint_actions, cost)
    
    cost = robot.net_time
    runtime = time.perf_counter() - start_time
    average_step_time /= count_steps
    gpaths = []
    for robot in [robot]:
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
    
    if print_pdf:
        plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.pdf')    
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')

    logfile = Path(args.save_dir) / f'results_{args.num_ugvs}UGV.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED: {args.seed} | UAVs: {args.num_drones} | PLANNER: {args.planner} | SUCC: {int(planning_exe.success)} | COST: {cost:0.3f} | T.TIME: {runtime:0.2f} | STEP.TIME : {average_step_time:0.2f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='sctp')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_ugvs', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=800)
    parser.add_argument('--C', type=float, default=300)
    parser.add_argument('--max_depth', type=int, default=50)
    parser.add_argument('--sampling_maps', type=int, default=80)
    parser.add_argument('--n_vertex', type=int, default=14)
    args = parser.parse_args()
    args.current_seed = args.seed

    _setup(args)
