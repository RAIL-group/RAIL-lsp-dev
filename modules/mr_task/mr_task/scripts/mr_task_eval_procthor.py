import numpy as np
import argparse
import procthor
from procthor.simulators import SceneGraphSimulator
import mr_task
import matplotlib.pyplot as plt
from common import Pose


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
        mrtask_planner.update(
            {'observed_graph': step_data['observed_graph']},
            step_data['robot_poses'],
            step_data['explored_container_nodes'],
            step_data['unexplored_container_nodes'],
            step_data['object_found'],
        )
        joint_action, cost = mrtask_planner.compute_joint_action()
        planning_loop.update_joint_action(joint_action)
    cost = min([robot.net_motion for robot in robot_team])
    fig = plt.figure(figsize=(10, 10))
    # plotting_grid = procthor.plotting.make_plotting_grid(np.transpose(known_grid))
    # plt.imshow(plotting_grid)
    procthor.plotting.plot_graph_on_grid(known_grid, known_graph)
    for robot in robot_team:
        path = np.array(robot.all_paths)
        plt.plot(path[0], path[1])
    plt.title(f'Seed: {args.seed} - Cost: {cost}')
    plt.savefig(f'{args.save_dir}/scene_graph_{args.seed}.png')
    plt.clf()
    plt.imshow(simulator.get_top_down_image())
    plt.savefig(f'{args.save_dir}/scene_graph_apt{args.seed}.png')

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
