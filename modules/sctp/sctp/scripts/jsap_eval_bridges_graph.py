import numpy as np
import argparse
from sctp.utils import plotting
import random, time
from sctp import sctp_graphs as graphs
from sctp.robot import Robot
from sctp import param, jsap
import matplotlib.pyplot as plt
from sctp.param import RobotType
from pathlib import Path
from sctp.planners import jsap_planner as planner
from sctp.planners import jsap_plan_exe as plan_loop


def _setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    print_pdf = False
    start, goal, graph = graphs.random_bridges_graph()
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    
    num_uav = 1
    num_ugv = 1
    assert args.num_drones == num_uav, "This script only supports 1 UAV"
    assert args.num_ugvs == num_ugv, "This script only supports 1 UGV"
    
    starts = [start]
    goals = [goal]
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(args.num_ugvs)]
    
    if args.planner =='ctp':
        args.num_drones = 0
        drones = []
        param.REVISIT_PEN = 2.0
        use_AVP = False
        use_DAP = False
        args.max_depth = 15
    elif args.planner =='jsap':
        args.num_drones = 1
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        assert args.num_drones == 1, "This script only supports 1 UAV"
        param.REVISIT_PEN = 0.0
        use_AVP = False
        use_DAP = False
        args.max_depth = 20
        
    elif args.planner == 'jsapiap':
        args.num_drones = 1
        param.REVISIT_PEN = 0.0
        args.max_depth = 20
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        use_AVP = True
        use_DAP = False
        assert args.num_drones == 1, "This script only supports 1 UAV"
    elif args.planner == 'jsapdap':
        args.num_drones = 1
        param.REVISIT_PEN = 0.0
        args.max_depth = 20
        drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    robot_type=RobotType.Drone, at_node=True) for i in range(args.num_drones)]
        use_AVP = False
        use_DAP = True
        assert args.num_drones == 1, "This script only supports 1 UAV"
    else:
        raise ValueError(f'Planner {args.planner} not recognized')

    assert args.num_iterations == 1000
    assert args.sampling_maps == 200
    print(f"Planner: {args.planner}, a team of {args.num_ugvs} UGV(s)-{args.num_drones} UAV(s), iters.: {args.num_iterations},"
          f" max depth: {args.max_depth}, maps: {args.sampling_maps}, AVP: {use_AVP}, DAP: {use_DAP}") 
    
     
    max_uanum = 1
    planner_robots = [robot.copy() for robot in robots]
    planner_drones = [drone.copy() for drone in drones]
    
    
    jsapplanner = planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, 
                                              uavs=planner_drones, rollout_fn=jsap.decsctp_rollout, C=args.C, 
                                              rollout_num=args.num_iterations, tree_depth=args.max_depth, n_maps=args.sampling_maps, 
                                              use_AVP=use_AVP, use_DAP=use_DAP, max_uanum=max_uanum, verbose=False)
    plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=jsapplanner.reached_goal, verbose=False)

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
        joint_actions, cost = jsapplanner.compute_joint_action()
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        plan_exec.save_joint_actions(joint_actions, cost)
    
    robot_net_times = [robot.net_time for robot in robots]
    cost_sum = np.sum(robot_net_times)
    cost_aver = np.average(robot_net_times)
    
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
                        seed=args.seed, cost=cost_sum, ttime=runtime, stime=average_step_time, verbose=True)
    
    if print_pdf:
        plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}_{args.num_drones}UAVs.pdf')    
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}_{args.num_drones}UAVs.png')

    logfile = Path(args.save_dir) / f'results_{args.num_ugvs}UGVs.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED: {args.seed} | UAVs: {args.num_drones} | PLANNER: {args.planner} | SUCC: {int(plan_exec.success)} "
                f"| COST_AVER: {cost_aver:0.3f} | COST_SUM: {cost_sum:0.3f} | T.TIME: {runtime:0.2f} | STEP.TIME : {average_step_time:0.2f} "
                f"| SAMP.TIME : {jsap.JSAPState.total_sampling_time:0.2f} | SPOLICY.TIME : {jsapplanner.single_policy_time:0.2f}\n")    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='sap')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_ugvs', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=800)
    parser.add_argument('--C', type=float, default=200)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--sampling_maps', type=int, default=80)
    parser.add_argument('--n_vertex', type=int, default=14)
    parser.add_argument('--max_uanum', type=int, default=1)
    # parser.add_argument('--spolicy_rollouts', type=int, default=300)
    args = parser.parse_args()
    args.current_seed = args.seed

    _setup(args)
