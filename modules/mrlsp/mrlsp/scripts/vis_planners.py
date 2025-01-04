import numpy as np
import environments
import lsp
import mrlsp
import matplotlib.pyplot as plt
from pathlib import Path


def vis_planners(args):
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    num_robots = args.num_robots
    start_poses, goal_poses = mrlsp.utils.utility.generate_start_and_goal(
        num_robots=num_robots,
        known_map=known_map,
        same_start=True,
        same_goal=True,
        def_start=pose,
        def_goal=goal)

    # Instantiate the simulated environment
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements,
        min_breadcrumb_signed_distance=4.0 * args.base_resolution)

    builder = environments.simulated.WorldBuildingUnityBridge

    # Create robots
    robot_team = mrlsp.utils.utility.get_robot_team(num_robots=num_robots,
                                                        start_poses=start_poses,
                                                        primitive_length=args.step_size,
                                                        num_primitives=args.num_primitives,
                                                        map_data=map_data)

    with builder(args.unity_path) as unity_bridge:
        unity_bridge.make_world(world)

        # create a simulator
        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             unity_bridge=unity_bridge,
                                             world=world)

        # set the inflation radius
        simulator.frontier_grouping_inflation_radius = simulator.inflation_radius

        # add the planner
        if args.planner == 'mrlsp':
            planner = mrlsp.planners.MRLearnedSubgoalPlanner(robot_team, goal_poses, args)
        elif args.planner == 'known':
            planner = mrlsp.planners.MRKnownSubgoalPlanner(robot_team, goal_poses, known_map, args)
        elif args.planner == 'optimistic':
            planner = mrlsp.planners.MROptimisticPlanner(robot_team, goal_poses, args)

        # planning loop
        planning_loop = mrlsp.planners.MRPlanningLoop(goal_poses,
                                                          known_map,
                                                          simulator,
                                                          unity_bridge=None,
                                                          robots=robot_team,
                                                          args=args)


        timestamp = 0

        for counter, step_data in enumerate(planning_loop):
            planner.update({'images': step_data['images']},
                           step_data['robot_grids'],
                           step_data['subgoals'],
                           step_data['robot_poses'],
                           step_data['visibility_masks'],
                           )
            joint_action = planner.compute_selected_subgoal()
            planning_loop.set_chosen_subgoals(joint_action, timestamp)

            paths = planning_loop.paths

            plt.ion()
            plt.figure(1)
            plt.clf()
            mrlsp.utils.plotting.visualize_mrlsp_result(
                plt.gca(),
                args,
                robot_team,
                planner.observed_map,
                planner.subgoals,
                goal_poses[0],
                paths,
                known_map)
            plt.show()
            plt.pause(0.1)
            image_filename = Path(args.save_dir)/'image.png'
            plt.savefig(image_filename, dpi=300)

            timestamp += 1

        cost = max(np.array(planning_loop.goal_reached) *
                   np.array([robot.net_motion for robot in robot_team]))
        print(f'Goal reached by robot {np.argmax(planning_loop.goal_reached)} with cost {cost:.2f} meters.')


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--num_robots', type=int)
    parser.add_argument('--comm_range', type=float, default=float('inf'))
    parser.add_argument('--planner', type=str, default='mrlsp')
    args = parser.parse_args()
    args.current_seed = args.seed[0]
    vis_planners(args)
