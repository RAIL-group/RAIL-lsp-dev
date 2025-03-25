import numpy as np
import matplotlib.pyplot as plt
import lsp
import procthor
from procthor.simulators import SceneGraphFrontierSimulator
import object_search
from object_search.planners import PlanningLoopPartialGrid, OptimisticFrontierPlanner, LearnedFrontierPlannerFCNN
from pathlib import Path


def object_search_eval(args):
    thor_interface = procthor.ThorInterface(args)
    known_graph, known_grid, robot_pose, target_obj_info = thor_interface.gen_map_and_poses()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / args.logfile_name
    with open(log_file, "a+") as f:
        f.write(f"LOG: {args.current_seed}\n")

    # Learned planner
    learned_simulator = SceneGraphFrontierSimulator(known_graph,
                                                    args,
                                                    target_obj_info,
                                                    known_grid,
                                                    thor_interface)

    learned_robot = lsp.robot.Turtlebot_Robot(robot_pose)
    learned_planner = LearnedFrontierPlannerFCNN(target_obj_info, args)
    learned_planning_loop = PlanningLoopPartialGrid(target_obj_info, learned_simulator, learned_robot, args=args)

    for _, step_data in enumerate(learned_planning_loop):
        learned_planner.update(
            step_data['observed_graph'],
            step_data['observed_grid'],
            step_data['containers'],
            step_data['frontiers'],
            step_data['robot_pose'])

        chosen_subgoal = learned_planner.compute_selected_subgoal()
        learned_planning_loop.set_chosen_subgoal(chosen_subgoal)

    learned_cost = learned_robot.net_motion
    learned_trajectory = np.array([[p.x, p.y] for p in learned_robot.all_poses]).T

    # Optimistic planner
    naive_simulator = SceneGraphFrontierSimulator(known_graph,
                                                  args,
                                                  target_obj_info,
                                                  known_grid,
                                                  thor_interface)
    naive_robot = lsp.robot.Turtlebot_Robot(robot_pose)
    naive_planner = OptimisticFrontierPlanner(target_obj_info, args)
    naive_planning_loop = PlanningLoopPartialGrid(target_obj_info, naive_simulator, naive_robot, args=args)

    for _, step_data in enumerate(naive_planning_loop):
        naive_planner.update(
            step_data['observed_graph'],
            step_data['observed_grid'],
            step_data['containers'],
            step_data['frontiers'],
            step_data['robot_pose'])

        chosen_subgoal = naive_planner.compute_selected_subgoal()
        naive_planning_loop.set_chosen_subgoal(chosen_subgoal)

    naive_cost = naive_robot.net_motion
    naive_trajectory = np.array([[p.x, p.y] for p in naive_robot.all_poses]).T

    did_succeed = learned_planning_loop.did_succeed and naive_planning_loop.did_succeed

    with open(log_file, "a+") as f:
        err_str = '' if did_succeed else '[ERR]'
        f.write(f"[Learn] {err_str} s: {args.current_seed:4d}"
                f" | learned: {learned_cost:0.3f}"
                f" | baseline: {naive_cost:0.3f}\n")

    plt.figure(figsize=(12, 6))
    known_locations = [known_graph.get_node_name_by_idx(idx) for idx in target_obj_info['container_idxs']]
    plt.suptitle(f"Seed: {args.current_seed} | Target object: {target_obj_info['name']}\n"
                 f"Known locations: {known_locations} ")

    ax = plt.subplot(221)
    top_down_image = thor_interface.get_top_down_image()
    plt.imshow(top_down_image)
    plt.title('Top down view of the scene')
    plt.axis('off')

    ax = plt.subplot(222)
    procthor.plotting.plot_graph_on_grid(ax, known_grid, known_graph)
    plt.text(robot_pose.x, robot_pose.y, '+', color='red', size=6, rotation=45)
    plt.title('Graph over known occupancy grid')

    ax = plt.subplot(223)
    object_search.plotting.plot_grid_with_robot_trajectory(ax, naive_planner.grid, naive_robot.all_poses,
                                                           naive_trajectory, naive_planner.graph)
    plt.title(f"Optimistic Cost: {naive_cost:0.3f}")

    ax = plt.subplot(224)
    object_search.plotting.plot_grid_with_robot_trajectory(ax, learned_planner.grid, learned_robot.all_poses,
                                                           learned_trajectory, learned_planner.graph)
    plt.title(f"Learned Cost: {learned_cost:0.3f}")

    plt.savefig(save_dir / f'object_search_frontier_{args.current_seed}.png', dpi=600)


if __name__ == '__main__':
    parser = object_search.utils.get_command_line_parser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--logfile_name', type=str, default='costs_log.txt')
    parser.add_argument('--image_filename', type=str)
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_file', type=str)
    args = parser.parse_args()

    object_search_eval(args)
