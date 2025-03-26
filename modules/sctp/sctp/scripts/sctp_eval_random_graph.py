import numpy as np
import argparse
import sctp
import random
from sctp import sctp_graphs as graphs
from sctp.robot import Robot
import matplotlib.pyplot as plt
from sctp.param import RobotType
from pathlib import Path
from sctp.planners import sctp_planner as planner
from sctp.planners import sctp_planning_loop as plan_loop


def _setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    start, goal, graph = graphs.disjoint_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        if poi.id == 6 or poi.id==8:
            poi.block_status = 0
    robot = Robot(position=[0.0, 0.0], cur_node=start.id, at_node=True)
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]

    # planner_graph = graph.copy()
    planner_robot = robot.copy()
    planner_drones = [drone.copy() for drone in drones]

    if args.planner == 'base':
        sctpplanner = planner.SCTPPlanner(args=args, init_graph=graph, goalID=goal.id,\
                                    robot=planner_robot, verbose=True) 
    
        planning_loop = plan_loop.SCTPPlanningLoop(robot=robot, goalID=goal.id,\
                                                   graph=graph, reached_goal=sctpplanner.reached_goal)

    elif args.planner == 'sctp':
        sctpplanner = planner.SCTPPlanner(args=args, init_graph=graph, goalID=goal.id,\
                                    robot=planner_robot, drones=planner_drones, verbose=True) 
    
        planning_loop = plan_loop.SCTPPlanningLoop(robot=robot, drones=drones, goalID=goal.id,\
                                                   graph=graph, reached_goal=sctpplanner.reached_goal)

    else:
        raise ValueError(f'Planner {args.planner} not recognized')
    
    
    for step_data in planning_loop:
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        
        joint_action, cost = sctpplanner.compute_joint_action()
        planning_loop.update_joint_action(joint_action, cost)
    
    cost = robot.net_time

    fig = plt.figure(figsize=(10, 10), dpi=300)
    # ax1 = plt.subplot(121)
    plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
    plt.text(start.coord[0]-0.5, start.coord[1], 'start', fontsize=7)
    plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
    plt.text(goal.coord[0]+0.1, goal.coord[1], 'goal', fontsize=7)
    graphs.plot_sctpgraph_combine(graph, plt)
    # print(robot.all_poses)
    x = [pose[0] for pose in robot.all_poses]
    y = [pose[1] for pose in robot.all_poses]
    plt.scatter(x, y, marker='P', s=4.5, alpha=1.0)
    plt.plot(x, y, color="green")
    for i, (x,y) in enumerate(zip(x,y)):
        xs = [x-0.1, x+0.1]
        ys = [y-0.1, y+0.15]
        plt.text(np.random.choice(xs),np.random.choice(ys), f'gs{i+1}', fontsize=5)
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        plt.plot(x,y, color='yellow', alpha=0.6)
        plt.scatter(x, y, marker='s', s=4.5)
        for i, (x,y) in enumerate(zip(x,y)):
            plt.text(x-0.1,y-0.2, f'ds{i+1}', fontsize=5)
    plt.title(f'Seed: {args.seed} | Planner: {args.planner} | Cost: {cost:.2f}')
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')

    # plt.show()

    logfile = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | COST : {cost:0.3f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='sctp')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=2000)
    parser.add_argument('--C', type=int, default=50)
    parser.add_argument('--resolution', type=float, default=0.05)
    args = parser.parse_args()
    args.current_seed = args.seed

    _setup(args)
