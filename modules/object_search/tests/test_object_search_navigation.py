import numpy as np
import matplotlib.pyplot as plt
import lsp
import procthor
from procthor.simulators import SceneGraphSimulator, SceneGraphFrontierSimulator
import object_search
from object_search.planners import PlanningLoop, OptimisticPlanner, PlanningLoopPartialGrid, OptimisticFrontierPlanner
from pathlib import Path


def get_args():
    args = lambda key: None  # noqa
    args.save_dir = '/data/test_logs'
    args.current_seed = 0
    args.resolution = 0.05
    args.inflation_radius_m = 0.0
    args.laser_max_range_m = 10.0
    args.disable_known_grid_correction = False
    args.laser_scanner_num_points = 1024
    args.field_of_view_deg = 360
    args.step_size = 1.8
    args.num_primitives = 32
    args.do_plot = True
    args.network_file = '/data/object_search/logs/FCNNforObjectSearch.pt'
    return args


def test_object_search_optimistic_planner():
    '''Test object search in ProcTHOR environment with OptimisticPlanner'''
    args = get_args()
    thor_interface = procthor.ThorInterface(args)
    known_graph, known_grid, robot_pose, target_obj_info = thor_interface.gen_map_and_poses()
    simulator = SceneGraphSimulator(known_graph,
                                    args,
                                    target_obj_info,
                                    known_grid,
                                    thor_interface)

    robot = object_search.robot.Robot(robot_pose)
    planner = OptimisticPlanner(target_obj_info, args)

    planning_loop = PlanningLoop(target_obj_info, simulator, robot, args=args, verbose=True)

    for _, step_data in enumerate(planning_loop):
        planner.update(
            step_data['observed_graph'],
            step_data['observed_grid'],
            step_data['subgoals'],
            step_data['robot_pose'])

        chosen_subgoal = planner.compute_selected_subgoal()
        planning_loop.set_chosen_subgoal(chosen_subgoal)

    cost, trajectory = object_search.utils.compute_cost_and_trajectory(known_grid, robot.all_poses)

    plt.figure(figsize=(8, 8))
    known_locations = [known_graph.get_node_name_by_idx(idx) for idx in target_obj_info['container_idxs']]
    plt.suptitle(f"Seed: {args.current_seed} | Target object: {target_obj_info['name']}\n"
                 f"Known locations: {known_locations} ")

    ax = plt.subplot(221)
    plt.title('Whole scene graph')
    procthor.plotting.plot_graph(ax, known_graph.nodes, known_graph.edges)

    ax = plt.subplot(222)
    procthor.plotting.plot_graph_on_grid(ax, known_grid, known_graph)
    plt.text(robot_pose.x, robot_pose.y, '+', color='red', size=6, rotation=45)
    plt.title('Graph over occupancy grid')

    plt.subplot(223)
    top_down_image = simulator.get_top_down_image()
    plt.imshow(top_down_image)
    plt.title('Top-down view of the map')
    plt.axis('off')

    plt.subplot(224)
    ax = plt.subplot(224)
    object_search.plotting.plot_grid_with_robot_trajectory(ax, known_grid, robot.all_poses, trajectory, known_graph)
    plt.title(f"Cost: {cost:0.3f}")

    plt.savefig(Path(args.save_dir) / f'object_search_optimistic_{args.current_seed}.png', dpi=1000)


def test_object_search_frontiers_optimistic():
    '''Test object search with frontiers and containers in ProcTHOR environment with OptimisticPlanner'''
    args = get_args()
    thor_interface = procthor.ThorInterface(args)
    known_graph, known_grid, robot_pose, target_obj_info = thor_interface.gen_map_and_poses()
    simulator = SceneGraphFrontierSimulator(known_graph,
                                            args,
                                            target_obj_info,
                                            known_grid,
                                            thor_interface)

    robot = lsp.robot.Turtlebot_Robot(robot_pose)
    planner = OptimisticFrontierPlanner(target_obj_info, args)

    planning_loop = PlanningLoopPartialGrid(target_obj_info, simulator, robot, args=args, verbose=True)
    plot_save_dir = Path(args.save_dir) / f'object_search_frontiers_optimistic_{args.current_seed}'
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    for counter, step_data in enumerate(planning_loop):
        planner.update(
            step_data['observed_graph'],
            step_data['observed_grid'],
            step_data['containers'],
            step_data['frontiers'],
            step_data['robot_pose'])

        chosen_subgoal = planner.compute_selected_subgoal()
        planning_loop.set_chosen_subgoal(chosen_subgoal)

        observed_grid = step_data['observed_grid']
        observed_graph = step_data['observed_graph']

        if args.do_plot:
            plt.ion()
            plt.clf()
            if isinstance(chosen_subgoal, object_search.core.Subgoal):
                title_text = f"Searching in container: {observed_graph.get_node_name_by_idx(chosen_subgoal.id)}"
            else:
                title_text = f"Exploring frontier at {chosen_subgoal.get_frontier_point()}"

            known_locations = [known_graph.get_node_name_by_idx(idx) for idx in target_obj_info['container_idxs']]
            plt.suptitle(f"Seed: {args.current_seed} | Target object: {target_obj_info['name']} | "
                         f"Known locations: {known_locations}\n"
                         f"{title_text}")

            ax = plt.subplot(121)
            procthor.plotting.plot_graph_on_grid(ax, known_grid, known_graph)
            plt.text(robot_pose.x, robot_pose.y, '+', color='red', size=6, rotation=45)
            plt.title('Known graph over known grid')

            ax = plt.subplot(122)
            procthor.plotting.plot_graph_on_grid(ax, observed_grid, observed_graph)
            plt.text(robot_pose.x, robot_pose.y, '+', color='red', size=6, rotation=45)
            plt.title('Observed graph over observed grid')

            cost = robot.net_motion
            trajectory = np.array([[p.x, p.y] for p in robot.all_poses]).T
            object_search.plotting.plot_grid_with_robot_trajectory(
                ax, observed_grid, robot.all_poses, trajectory, observed_graph
            )
            if planning_loop.current_path is not None:
                ax.plot(planning_loop.current_path[0, :], planning_loop.current_path[1, :], 'b:')
            plt.title(f"Cost: {cost:0.3f}")
            # plt.savefig(plot_save_dir / f'plan_{args.current_seed}_{counter}.png', dpi=1000)
            plt.show()
            plt.pause(0.01)

    cost = robot.net_motion
    trajectory = np.array([[p.x, p.y] for p in robot.all_poses]).T

    plt.figure(figsize=(8, 8), dpi=1000)
    known_locations = [known_graph.get_node_name_by_idx(idx) for idx in target_obj_info['container_idxs']]
    plt.suptitle(f"Seed: {args.current_seed} | Target object: {target_obj_info['name']}\n"
                 f"Known locations: {known_locations} ")

    ax = plt.subplot(221)
    plt.title('Whole scene graph')
    procthor.plotting.plot_graph(ax, known_graph.nodes, known_graph.edges)

    ax = plt.subplot(222)
    procthor.plotting.plot_graph_on_grid(ax, known_grid, known_graph)
    plt.text(robot_pose.x, robot_pose.y, '+', color='red', size=6, rotation=45)
    plt.title('Graph over occupancy grid')

    plt.subplot(223)
    top_down_image = simulator.get_top_down_image()
    plt.imshow(top_down_image)
    plt.title('Top-down view of the map')
    plt.axis('off')

    ax = plt.subplot(224)
    object_search.plotting.plot_grid_with_robot_trajectory(ax, known_grid, robot.all_poses, trajectory, known_graph)
    plt.title(f"Cost: {cost:0.3f}")

    plt.savefig(Path(args.save_dir) / f'object_search_frontiers_optimistic_{args.current_seed}.png', dpi=1000)
