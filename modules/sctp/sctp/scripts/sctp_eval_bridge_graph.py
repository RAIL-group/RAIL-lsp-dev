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

# def sgraph_init():
#     start, goal, graph = graphs.s_graph_unc()
#     for poi in graph.pois:
#         assert poi.block_prob != 0.0
#         assert poi.block_prob != 1.0
#         if poi.id == 6 or poi.id==8 or poi.id==7:
#             poi.block_status = int(0)
#         if poi.id == 5 or poi.id==9:
#             poi.block_status = int(1)
#     return start, goal, graph

# def mgraph_init():
#     start, goal, graph = graphs.m_graph_unc()
#     for poi in graph.pois:
#         assert poi.block_prob != 0.0
#         assert poi.block_prob != 1.0
#         if poi.id == 8 or poi.id==9 or poi.id==14 or poi.id==19 or poi.id==18:
#             poi.block_status = int(0)
#     return start, goal, graph


def _setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    need_pdf = False
    # print("Generating random bridges graph")
    start, goal, graph = graphs.random_bridges_graph()
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    planner_robot = robot.copy()
    
    # param.IV_SAMPLE_SIZE = 1000
    if args.planner == 'base':
        drones = []
        args.num_drones = 0
        param.ADD_IV = False
        param.REVISIT_PEN = 6.0
    elif args.planner =='jsctp1':
        args.num_drones = 1
        param.ADD_IV = False
        drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]
    elif args.planner == 'sctpiv':
        drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]
        param.ADD_IV = True
    else:
        raise ValueError(f'Planner {args.planner} not recognized')
    
    planner_drones = [drone.copy() for drone in drones]

    sctpplanner = planner.SCTPPlanner(init_graph=policyGraph, goalID=goal.id, robot=planner_robot, 
                                        drones=planner_drones, tree_depth=args.max_depth, C=args.C,
                                        rollout_num=args.num_iterations,
                                      n_maps=args.sampling_maps, rollout_fn=core.sctp_rollout3) 
    planning_exe = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goal.id,\
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
    goals_cords = [goal.coord for goal in [goal]]
    starts_cords = [start.coord for start in [start]]
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpaths=gpaths, dpaths=dpaths, \
                    graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
                        seed=args.seed, cost=cost, verbose=True)
    if need_pdf:
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
    parser.add_argument('--num_iterations', type=int, default=1500)
    parser.add_argument('--C', type=float, default=50)
    parser.add_argument('--max_depth', type=int, default=100)
    parser.add_argument('--sampling_maps', type=int, default=100)
    args = parser.parse_args()
    args.current_seed = args.seed

    _setup(args)
