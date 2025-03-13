import numpy as np
import argparse
import procthor
from procthor.simulators import SceneGraphSimulator
import mr_task
import matplotlib.pyplot as plt
from common import Pose
from pathlib import Path


def _setup(args):
    thor_interface = procthor.ThorInterface(args)
    known_graph, known_grid, robot_pose, target_objs_info = thor_interface.gen_map_and_poses(num_objects=3)
    robot_team = [mr_task.robot.Robot(robot_pose) for _ in range(args.num_robots)]

    specification = mr_task.specification.get_random_specification(objects=[obj['name'] for obj in target_objs_info],
                                                                   seed=args.current_seed)
    print(specification)

    if args.planner == 'optimistic':
        mrtask_planner = mr_task.planner.OptimisticMRTaskPlanner(args, specification)
    elif args.planner == 'learned':
        mrtask_planner = mr_task.planner.LearnedMRTaskPlanner(args, specification)
    else:
        raise ValueError(f'Planner {args.planner} not recognized')

    simulator = SceneGraphSimulator(args=args, known_graph=known_graph,
                    target_objs_info=target_objs_info, known_grid=known_grid, thor_interface=thor_interface)

    planning_loop = mr_task.planner.MRTaskPlanningLoop(
        robot_team, simulator, mrtask_planner.dfa_planner.has_reached_accepting_state)


    for step_data in planning_loop:
        print("Total containers to left explore: ", len(planning_loop.containers_idx))
        mrtask_planner.update(
            {'observed_graph': step_data['observed_graph'],
             'observed_map': step_data['observed_map']},
            step_data['robot_poses'],
            step_data['explored_container_nodes'],
            step_data['unexplored_container_nodes'],
            step_data['object_found'],
        )
        joint_action, cost = mrtask_planner.compute_joint_action()
        planning_loop.update_joint_action(joint_action)

    cost = min([robot.net_motion for robot in robot_team])

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax1 = plt.subplot(121)
    procthor.plotting.plot_graph_on_grid(known_grid, known_graph)
    plt.scatter(robot_pose.x, robot_pose.y, marker='o', color='r')
    plt.text(robot_pose.x, robot_pose.y, 'start', fontsize=8)
    for robot in robot_team:
        path = np.array(robot.all_paths)
        plt.plot(path[0], path[1])
        x = [pose.x for pose in robot.all_poses]
        y = [pose.y for pose in robot.all_poses]
        ax1.scatter(x, y, marker='x', alpha=0.5)
    plt.title(f'Seed: {args.seed} | Planner: {args.planner} | Cost: {cost:.2f}')
    ax2 = plt.subplot(122)
    procthor.plotting.plot_graph(ax2, known_graph.nodes, known_graph.edges)
    plt.suptitle(f'Specification: {specification}', fontweight="bold")
    ordering = str(', '.join([str(node) for node in planning_loop.ordering]))
    fig.supxlabel(f'Visit order: {ordering}', wrap=True)
    plt.savefig(f'{args.save_dir}/mtask_eval_planner_{args.planner}_seed_{args.seed}.png')

    logfile = Path(args.save_dir) / f'log_{args.num_robots}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | COST : {cost:0.3f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/mr_task')
    parser.add_argument('--network_file', type=str, default='/data/mr_task')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='optimistic')
    parser.add_argument('--num_robots', type=int, default=2)
    parser.add_argument('--num_iterations', type=int, default=50000)
    parser.add_argument('--C', type=int, default=100)
    parser.add_argument('--resolution', type=float, default=0.05)
    args = parser.parse_args()
    args.current_seed = args.seed

    _setup(args)
