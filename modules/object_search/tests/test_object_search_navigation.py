import matplotlib.pyplot as plt
import procthor
from procthor.simulators import SceneGraphSimulator
import object_search
from object_search.planners import PlanningLoop, OptimisticPlanner
from pathlib import Path


def get_args():
    args = lambda key: None  # noqa
    args.save_dir = '/data/test_logs'
    args.current_seed = 0
    args.resolution = 0.05
    args.do_save_video = False
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

    cost, trajectory = object_search.utils.compute_cost_and_trajectory(known_grid, robot.all_poses, args.resolution,
                                                                       use_robot_model=True)

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
    plt.title(f"Cost: {cost:0.1f} meters")

    plt.savefig(Path(args.save_dir) / f'object_search_optimistic_{args.current_seed}.png', dpi=1000)

    video_file_path = Path(args.save_dir) / f'object_search_optimistic_{args.current_seed}.mp4'
    fig_title = f'Seed: {args.current_seed} | Target object: {target_obj_info["name"]}'

    if args.do_save_video:
        object_search.plotting.save_navigation_video(trajectory,
                                                     thor_interface,
                                                     video_file_path=video_file_path,
                                                     fig_title=fig_title)
