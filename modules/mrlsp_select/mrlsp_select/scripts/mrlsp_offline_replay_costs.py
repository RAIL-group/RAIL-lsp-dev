import numpy as np
import matplotlib.pyplot as plt
import environments
import lsp
import lsp_select
from mrlsp.planners import MRLearnedSubgoalPlanner, MROptimisticPlanner
from mrlsp_select.planners import MRPolicySelectionPlanner
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
                plt.ion()
                plt.figure(1)
                plt.clf()
                mrlsp.utils.plotting.visualize_robots(args,
                                                      robot_team,
                                                      planner.observed_map,
                                                      step_data['images'],
                                                      planner.subgoals,
                                                      goal_poses[0],
                                                      planning_loop.paths,
                                                      known_map,
                                                      timestamp=None)
                plt.show()
                plt.pause(0.1)
    # First robot to reach goal
    robot_idx = planning_loop.goal_reached.index(True) # noqa
    mrlsp.utils.plotting.visualize_robots(args,
                                              robot_team,
                                              planner.observed_map,
                                              step_data['images'],
                                              planner.subgoals,
                                              goal_poses[0],
                                              planning_loop.paths,
                                              known_map,
                                              timestamp=None)
    plt.suptitle(f'{args.chosen_planner} ({args.env})')
    plt.xlabel(f'seed={args.current_seed}')
    plt.savefig(Path(args.save_dir) / f'img_{args.chosen_planner}_{args.env}_{args.current_seed}.png', dpi=200)

    return planner, robot_team[robot_idx]


if __name__ == "__main__":
    maze_params = {
        'envs': ['envA', 'envB', 'envC'],
        'planners': ['nonlearned', 'lspA', 'lspB', 'lspC'],
        'network_files': [None, 'mazeA/mazeA.pt', 'mazeB/mazeB.pt', 'mazeC/mazeC.pt']
    }
    office_params = {
        'envs': ['mazeA', 'office', 'officewall'],
        'planners': ['nonlearned', 'lspmaze', 'lspoffice', 'lspofficewallswap'],
        'network_files': [None, 'mazeA/mazeA.pt', 'office/office_base.pt', 'office_wallswap/office_wallswap.pt']
    }
    env_params = {
        'maze': maze_params,
        'office': office_params
    }

    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--experiment_type', choices=['maze', 'office'])
    args, _ = parser.parse_known_args()
    EXPERIMENT_TYPE = args.experiment_type
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_path', type=str)
    parser.add_argument('--env', choices=env_params[EXPERIMENT_TYPE]['envs'])
    parser.add_argument('--chosen_planner', choices=env_params[EXPERIMENT_TYPE]['planners'])
    parser.add_argument('--num_robots', type=int)
    parser.add_argument('--comm_range', type=float, default=float('inf'))
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--ucb_c', type=float, default=500)
    args = parser.parse_args()

    args.planner_names = env_params[EXPERIMENT_TYPE]['planners']
    args.network_files = env_params[EXPERIMENT_TYPE]['network_files']

    all_planners = '_'.join(args.planner_names)

    args.current_seed = args.seed[0]
    path = Path(args.save_dir)
    cost_file = path / f'cost_{args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.txt'
    err_file = path / f'error_{args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.txt'
    lb_costs_file = path / f'lbc_{args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.txt'
    target_file = path / f'target_plcy_{args.chosen_planner}_envrnmnt_{args.env}_{args.current_seed}.txt'

    if cost_file.is_file():
        print(f'Data already exists for {args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.')
        exit()
    if err_file.is_file():
        print(f'Error file exists for {args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.')
        exit()

    print(f'Generating data for {args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.')

    planner, robot = maze_eval(args)
    try:
        costs, lb_costs = planner.get_costs(robot)

        with open(cost_file, 'w') as f:
            np.savetxt(f, costs)
        with open(lb_costs_file, 'w') as f:
            np.savetxt(f, lb_costs)
        with open(target_file, 'w') as f:
            f.write('\n')
    except TypeError as e:
        with open(err_file, 'w') as f:
            f.write(f'{e}')
            print(f'{e}')
        with open(target_file, 'w') as f:
            f.write('\n')
