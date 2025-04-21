import numpy as np
import environments
import lsp
import mrlsp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def mrlsp_evaluate(args):
    logfile = f'{args.logfile_dir}_robots_{args.num_robots}.txt'
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
    robot_team_optimistic = mrlsp.utils.utility.get_robot_team(num_robots=num_robots,
                                                                   start_poses=start_poses,
                                                                   primitive_length=args.step_size,
                                                                   num_primitives=args.num_primitives,
                                                                   map_data=map_data)

    robot_team_mrlsp = mrlsp.utils.utility.get_robot_team(num_robots=num_robots,
                                                              start_poses=start_poses,
                                                              primitive_length=args.step_size,
                                                              num_primitives=args.num_primitives,
                                                              map_data=map_data)
    with builder(args.unity_path) as unity_bridge:
        unity_bridge.make_world(world)

        # create a simulator
        simulator_optimistic = lsp.simulators.Simulator(known_map,
                                                        goal,
                                                        args,
                                                        unity_bridge=unity_bridge,
                                                        world=world)

        simulator_mrlsp = lsp.simulators.Simulator(known_map,
                                                   goal,
                                                   args,
                                                   unity_bridge=unity_bridge,
                                                   world=world)

        # set the inflation radius
        simulator_optimistic.frontier_grouping_inflation_radius = simulator_optimistic.inflation_radius
        simulator_mrlsp.frontier_grouping_inflation_radius = simulator_mrlsp.inflation_radius

        # add the planner
        planner_optimistic = mrlsp.planners.MROptimisticPlanner(robot_team_optimistic, goal_poses, args)
        planner_mrlsp = mrlsp.planners.MRLearnedSubgoalPlanner(robot_team_mrlsp, goal_poses, args)

        # planning loop
        planning_loop_optimistic = mrlsp.planners.MRPlanningLoop(goal_poses,
                                                                     known_map,
                                                                     simulator_optimistic,
                                                                     unity_bridge=None,
                                                                     robots=robot_team_optimistic,
                                                                     args=args)

        planning_loop_mrlsp = mrlsp.planners.MRPlanningLoop(goal_poses,
                                                                known_map,
                                                                simulator_mrlsp,
                                                                unity_bridge=None,
                                                                robots=robot_team_mrlsp,
                                                                args=args)
        timestamp_optimistic = 0
        timestamp_mrlsp = 0

        for counter, step_data in enumerate(planning_loop_optimistic):
            planner_optimistic.update({'images': step_data['images']},
                                      step_data['robot_grids'],
                                      step_data['subgoals'],
                                      step_data['robot_poses'],
                                      step_data['visibility_masks'],
                                      )
            joint_action = planner_optimistic.compute_selected_subgoal()
            planning_loop_optimistic.set_chosen_subgoals(joint_action, timestamp_optimistic)
            paths_optimistic = planning_loop_optimistic.paths
            timestamp_optimistic += 1
        optimistic_cost = max(np.array(planning_loop_optimistic.goal_reached) *
                              np.array([robot.net_motion for robot in robot_team_optimistic]))

        for counter, step_data in enumerate(planning_loop_mrlsp):
            planner_mrlsp.update({'images': step_data['images']},
                                 step_data['robot_grids'],
                                 step_data['subgoals'],
                                 step_data['robot_poses'],
                                 step_data['visibility_masks'],
                                 )
            joint_action = planner_mrlsp.compute_selected_subgoal()
            planning_loop_mrlsp.set_chosen_subgoals(joint_action, timestamp_mrlsp)
            paths_mrlsp = planning_loop_mrlsp.paths
            timestamp_mrlsp += 1

        mrlsp_cost = max(np.array(planning_loop_mrlsp.goal_reached) *
                         np.array([robot.net_motion for robot in robot_team_mrlsp]))

        plt.ion()
        plt.figure(figsize=(8, 4), dpi=300)
        gs = GridSpec(1, 2)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        mrlsp.utils.plotting.visualize_mrlsp_result(
            ax1,
            args,
            robot_team_optimistic,
            planner_optimistic.observed_map,
            planner_optimistic.subgoals,
            goal_poses[0],
            paths_optimistic,
            known_map)
        ax1.set_title(f'Optimistic cost: {optimistic_cost:.2f}')

        mrlsp.utils.plotting.visualize_mrlsp_result(
            ax2,
            args,
            robot_team_mrlsp,
            planner_mrlsp.observed_map,
            planner_mrlsp.subgoals,
            goal_poses[0],
            paths_mrlsp,
            known_map)
        ax2.set_title(f'MRLSP cost: {mrlsp_cost:.2f}')

        plt.savefig(f'{args.save_dir}/mrlsp_eval_{args.seed[0]}_r{args.num_robots}.png')

        with open(logfile, "a+") as f:
            if np.all(planning_loop_mrlsp.is_stuck):
                f.write(f"SEED : {args.seed[0]}"
                        f" | FAILED: MRLSP both robot stuck\n")
                return
            f.write(f"SEED : {args.seed[0]}"
                    f" | learned: {mrlsp_cost:0.3f}"
                    f" | optimistic: {optimistic_cost:0.3f}\n")


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--num_robots', type=int)
    parser.add_argument('--comm_range', type=float, default=float('inf'))
    parser.add_argument('--logfile_dir', type=str)
    args = parser.parse_args()
    args.current_seed = args.seed[0]
    mrlsp_evaluate(args)
