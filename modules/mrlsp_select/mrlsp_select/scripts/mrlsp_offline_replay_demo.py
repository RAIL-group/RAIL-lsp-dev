import numpy as np
import matplotlib.pyplot as plt
import environments
import lsp
import lsp_select
from mrlsp.planners import MRLearnedSubgoalPlanner, MROptimisticPlanner
from mrlsp_select.planners import MRLSPInfoGatherPlanner, MRPolicySelectionPlanner
# from lsp_select.utils.misc import corrupt_robot_pose
from pathlib import Path
import mrlsp


def maze_eval(args):
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

        # # add the planner
        # planner = mrlsp.planners.MROptimisticPlanner(args, robot_team, goal_poses)
        # planner = mrlsp.planners.MRLearnedSubgoalPlanner(robot_team, goal_pose, args)

        # planning loop
        planning_loop = mrlsp.planners.MRPlanningLoop(goal_poses,
                                                      known_map,
                                                      simulator,
                                                      unity_bridge=None,
                                                      robots=robot_team,
                                                      args=args)

        # args.robot = robot
        args.robot_pose = pose
        args.map_shape = known_map.shape
        # args.known_map = known_map

        planners = []
        for network in args.network_files:
            if network is None:
                planners.append(MROptimisticPlanner(robot_team, goal_poses, args))
            else:
                args.network_file = str(Path(args.network_path) / network)
                planners.append(MRLearnedSubgoalPlanner(robot_team, goal_poses, args))
        chosen_planner_idx = args.planner_names.index(args.chosen_planner)

        planner = MRPolicySelectionPlanner(goal, planners, chosen_planner_idx, args)

        for counter, step_data in enumerate(planning_loop):
            planner.update(
                {'images': step_data['images']},
                step_data['robot_grids'],
                step_data['subgoals'],
                step_data['robot_poses'],
                step_data['visibility_masks'])
            planning_loop.set_chosen_subgoals(planner.compute_selected_subgoal())

            if args.do_plot:
                paths = planning_loop.paths
                plt.ion()
                plt.figure(1)
                plt.clf()
                mrlsp.utils.plotting.visualize_robots(args,
                                                      robot_team,
                                                      planner.observed_map,
                                                      step_data['images'],
                                                      planner.subgoals,
                                                      goal_poses[0],
                                                      paths,
                                                      known_map,
                                                      timestamp=None)
                # for f in planner.alt_subgoal:
                #     plt.scatter(*f.get_centroid())
                plt.show()
                plt.pause(0.1)
    # First robot to reach goal
    robot_idx = planning_loop.goal_reached.index(True) # noqa

    return planner, robot_team[robot_idx]


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_path', type=str)
    parser.add_argument('--env', choices=['envA', 'envB', 'envC'])
    parser.add_argument('--chosen_planner', choices=['nonlearned', 'lspA', 'lspB', 'lspC'])
    parser.add_argument('--num_robots', type=int)
    parser.add_argument('--comm_range', type=float, default=float('inf'))
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--ucb_c', type=float, default=500)
    args = parser.parse_args()
    args.current_seed = args.seed[0]

    if args.env == 'mazeA' and args.map_type != 'maze':
        raise ValueError('map_type should be "maze" when env is "mazeA"')

    args.planner_names = ['nonlearned', 'lspA', 'lspB', 'lspC']
    args.network_files = [None, 'mazeA/mazeA.pt', 'mazeB/mazeB.pt', 'mazeC/mazeC.pt']

    planner, robot = maze_eval(args)
    costs, lb_costs = planner.get_costs(robot)
    print(costs)
    print(lb_costs)
    cost_file = Path(args.save_dir) / f'cost_{args.chosen_planner}_{args.env}_{args.current_seed}.txt'
    lb_costs_file = Path(args.save_dir) / f'lbc_{args.chosen_planner}_{args.env}_{args.current_seed}.txt'
    with open(cost_file, 'w') as f:
        np.savetxt(f, costs)
    with open(lb_costs_file, 'w') as f:
        np.savetxt(f, lb_costs)
