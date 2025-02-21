import os
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import procthor
import taskplan
from taskplan.planners import (
    ClosestActionPlanner,
    LearnedPlanner,
    KnownPlanner
)
from taskplan_select.planners import LSPLLMPlanner
from taskplan_select.simulators import SceneGraphSimulator
from pathlib import Path


def evaluate_main(args):
    # Load data for a given seed
    thor_interface = procthor.ThorInterface(args=args)
    known_graph, known_grid, robot_pose, target_obj_info = thor_interface.gen_map_and_poses()
    # print(target_obj_info)
    # exit()
    # Initialize the PartialMap with whole graph
    # partial_map = taskplan.core.PartialMap(known_graph, known_grid)
    simulator = SceneGraphSimulator(known_graph,
                                    args,
                                    target_obj_info,
                                    known_grid,
                                    thor_interface)

    # Intialize logfile
    logfile = os.path.join(args.save_dir, args.logfile_name)
    with open(logfile, "a+") as f:
        f.write(f"LOG: {args.current_seed}\n")

    ################
    # ~~~ Naive ~~ #
    ################
    naive_robot = taskplan.robot.Robot(robot_pose)
    naive_planner = ClosestActionPlanner(target_obj_info, args)
    naive_cost_str = 'naive'
    naive_planning_loop = taskplan.planners.planning_loop.PlanningLoop(
        target_obj_info, simulator, robot=naive_robot, args=args,
        verbose=True)

    for counter, step_data in enumerate(naive_planning_loop):
        # Update the planner objects
        naive_planner.update(
            step_data['observed_graph'],
            step_data['observed_grid'],
            step_data['subgoals'],
            step_data['robot_pose'])

        # Compute the next subgoal and set to the planning loop
        chosen_subgoal = naive_planner.compute_selected_subgoal()
        naive_planning_loop.set_chosen_subgoal(chosen_subgoal)

    naive_path = naive_robot.all_poses
    naive_dist, naive_trajectory = taskplan.core.compute_path_cost(known_grid, naive_path)

    ################
    # ~~ Learned ~ #
    ################
    # learned_planner = LearnedPlanner(target_obj_info, args)
    # learned_planner = LSPLLMPlanner(target_obj_info, args)
    learned_robot = taskplan.robot.Robot(robot_pose)
    learned_planner = LSPLLMPlanner(target_obj_info, args, fake_llm_response_text='100%')
    # learned_planner =['nodes'] LearnedPlanner(args, partial_map)
    learned_cost_str = 'learned'
    learned_planning_loop = taskplan.planners.PlanningLoop(
        target_obj_info, simulator, robot=learned_robot, args=args,
        verbose=True)

    for counter, step_data in enumerate(learned_planning_loop):
        # Update the planner objects
        learned_planner.update(
            step_data['observed_graph'],
            step_data['observed_grid'],
            step_data['subgoals'],
            step_data['robot_pose'])

        # Compute the next subgoal and set to the planning loop
        chosen_subgoal = learned_planner.compute_selected_subgoal()
        learned_planning_loop.set_chosen_subgoal(chosen_subgoal)

        plt.ion()
        plt.figure(1, figsize=(5, 5))
        plt.clf()
        ax = plt.subplot(111)
        plt.title(f"Chosen Subgoal: {step_data['observed_graph'].get_node_name_by_idx(chosen_subgoal.id)}")
        procthor.plotting.plot_graph(ax, step_data['observed_graph'].nodes, step_data['observed_graph'].edges,
                                     highlight_node=chosen_subgoal.id)
        # procthor.plotting.plot_graph(ax, graph['nodes'], graph['edge_index'])
        plt.savefig(Path(args.save_dir) / f'path_{args.current_seed}_{counter}.png', dpi=600)

    learned_path = learned_robot.all_poses
    learned_dist, learned_trajectory = taskplan.core.compute_path_cost(known_grid, learned_path)
    with open(logfile, "a+") as f:
        f.write(f"[Evaluation] s: {args.current_seed:4d}"
                f" | naive: {naive_dist:0.3f}"
                f" | learned: {learned_dist:0.3f}\n"
                f"  Steps: {len(naive_path)-1:3d}"
                f" | {len(learned_path)-1:3d}\n")
    make_combined_plot(args, known_graph, known_grid, robot_pose, target_obj_info,
                       naive_cost_str, naive_path, naive_dist, naive_trajectory,
                       learned_cost_str, learned_path, learned_dist, learned_trajectory,
                       simulator.get_top_down_image())


def make_combined_plot(args, known_graph, known_grid, robot_pose, target_obj_info,
                       naive_cost_str, naive_path, naive_dist, naive_trajectory,
                       learned_cost_str, learned_path, learned_dist, learned_trajectory,
                       top_down_frame):
    # plt.clf()
    plt.figure(figsize=(10, 5))
    what = target_obj_info['name']
    where = [known_graph.get_node_name_by_idx(goal) for goal in target_obj_info['container_idx']]
    plt.suptitle(f"Find {what} from {where} in seed: [{args.current_seed}]", fontsize=9)

    ax = plt.subplot(231)
    # 1 plot the whole graph
    plt.title('Whole scene graph', fontsize=6)
    procthor.plotting.plot_graph(ax, known_graph.nodes, known_graph.edges)
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])
    ######################

    plt.subplot(232)
    # 2 plot the graph overlaied image
    procthor.plotting.plot_graph_on_grid(known_grid, known_graph)
    x, y = robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title('Graph over occupancy grid', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    ######################

    plt.subplot(233)
    # 3 plot the top-dwon-view
    plt.imshow(top_down_frame)
    plt.title('Top-down view of the map', fontsize=6)
    plt.axis('off')

    ######################

    plt.subplot(234)
    # 4 plot the grid with naive trajectory viridis color
    plotting_grid = procthor.plotting.make_plotting_grid(
        np.transpose(known_grid)
    )
    plt.imshow(plotting_grid)
    plt.title(f"{naive_cost_str} Cost: {naive_dist:0.3f}", fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    x, y = naive_path[0]
    plt.text(x, y, '0 - ROBOT', color='brown', size=4)

    for idx, coords in enumerate(naive_path[1:]):
        # find the node_idx for this pose and use it through
        # graph['node_coords']
        pose = taskplan.utilities.utils. \
            get_pose_from_coord(coords, known_graph.nodes)
        name = known_graph.get_node_name_by_idx(pose)
        # x, y = known_graph.get_node_position_by_idx(pose)
        x, y = coords
        plt.text(x, y, f'{idx+1} - {pose}:{name}', color='brown', size=4)

    # Create a Viridis color map
    viridis_cmap = plt.get_cmap('viridis')
    # Generate colors based on the Viridis color map
    colors = np.linspace(0, 1, len(naive_trajectory[0]))
    line_colors = viridis_cmap(colors)

    # Plot the points with Viridis color gradient
    for idx, x in enumerate(naive_trajectory[0]):
        y = naive_trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)
    ######################

    plt.subplot(235)
    # 4 plot the grid with learned trajectory viridis color
    # plotting_grid = procthor.plotting.make_plotting_grid(
    #     np.transpose(grid)
    # )
    plt.imshow(plotting_grid)
    plt.title(f"{learned_cost_str} Cost: {learned_dist:0.3f}", fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    x, y = learned_path[0]
    plt.text(x, y, '0 - ROBOT', color='brown', size=4)

    for idx, coords in enumerate(learned_path[1:]):
        # # find the node_idx for this pose and use it through
        pose = taskplan.utilities.utils. \
            get_pose_from_coord(coords, known_graph.nodes)
        name = known_graph.get_node_name_by_idx(pose)
        # x, y = known_graph.get_node_position_by_idx(pose)
        x, y = coords
        plt.text(x, y, f'{idx+1} - {pose}:{name}', color='brown', size=4)
    # Create a Viridis color map
    viridis_cmap = plt.get_cmap('viridis')
    # Generate colors based on the Viridis color map
    colors = np.linspace(0, 1, len(learned_trajectory[0]))
    line_colors = viridis_cmap(colors)

    # Plot the points with Viridis color gradient
    for idx, x in enumerate(learned_trajectory[0]):
        y = learned_trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)
    ######################

    plt.savefig(Path(args.save_dir) / f'combined_{args.current_seed}.png', dpi=1000)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for Object Search"
    )
    parser.add_argument('--current_seed', type=int, required=True)
    parser.add_argument('--logfile_name', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--resolution', type=float, required=True)
    parser.add_argument('--network_file', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    evaluate_main(args)
