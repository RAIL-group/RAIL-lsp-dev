import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

import procint
import procthor
import taskplan


def intervention_pipeline(args, do_intervene=True):
    # Load data for a given seed
    thor_data = procthor.procthor.ThorInterface(args=args)

    # Get the occupancy grid from data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph, grid)
    partial_map.set_room_info(init_robot_pose, thor_data.rooms)
    # partial_map.target_obj = 9

    planning_loop = taskplan.planners.planning_loop.PlanningLoop(
        partial_map=partial_map, robot=init_robot_pose,
        destination=None, args=args, verbose=True)

    planner = procint.planners.IntvPlanner(args, partial_map, verbose=True)

    known_planner = taskplan.planners.planner.KnownPlanner(args, partial_map)

    agrees_with_oracle = []

    for counter, step_data in enumerate(planning_loop):
        # Update the planner objects
        planner.update(
            step_data['graph'],
            step_data['subgoals'],
            step_data['robot_pose'])

        known_planner.update(
            step_data['graph'],
            step_data['subgoals'],
            step_data['robot_pose'])

        # Compute the subgoal prefered by learned planner
        chosen_subgoal = planner.compute_selected_subgoal()
        # Identify the subgoal leading to the goal from known planner
        target_subgoal = known_planner.compute_selected_subgoal()

        # Say whether the two planners agree
        agrees_with_oracle.append(chosen_subgoal == target_subgoal)

        # Intervene and generate counter factual explanation with target subgoal
        # without freezeing the planner chosen subgoal and updating the planner
        if do_intervene and counter == args.intervene_at and \
           chosen_subgoal != target_subgoal:
            print('Intervention invoked!')
            planner.generate_counterfactual_explanation(
                target_subgoal, do_freeze_selected=False, keep_changes=True,
                margin=5, learning_rate=args.learning_rate)
            chosen_subgoal = planner.compute_selected_subgoal()

        planning_loop.set_chosen_subgoal(chosen_subgoal)

    cost_str = 'after intervention' if do_intervene else 'before'
    image_file = os.path.join(
        args.save_dir, 'images',
        f'intervention_seed_{args.current_seed}_{int(do_intervene)}.png')
    path = planning_loop.robot
    dist, trajectory = taskplan.core.compute_path_cost(partial_map.grid, path)

    what = partial_map.org_node_names[partial_map.target_obj]
    where = [partial_map.org_node_names[goal] for goal in partial_map.target_container]

    plt.clf()
    plt.figure(figsize=(8, 4))
    plt.suptitle(f"Find {what} from {where} in seed: [{args.current_seed}]", fontsize=9)

    # plot the underlying graph (except objects) on grid
    plt.subplot(121)
    plt.title("Underlying grid with containers", fontsize=6)
    procthor.plotting.plot_graph_on_grid_old(partial_map.grid, whole_graph)
    # plt.axis('equal')
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    x, y = path[0]
    plt.text(x, y, '+', color='red', size=6, rotation=45)

    # plot the trajectory
    plt.subplot(122)
    plotting_grid = procthor.plotting.make_plotting_grid(
        np.transpose(partial_map.grid)
    )
    plt.title(f"{cost_str}: {dist:0.3f}", fontsize=6)
    plt.imshow(plotting_grid)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    x, y = path[0]
    plt.text(x, y, '0 - ROBOT', color='brown', size=4)

    for idx, coords in enumerate(path[1:]):
        # find the node_idx for this pose and use it through
        # graph['node_coords']
        pose = taskplan.utilities.utils. \
            get_pose_from_coord(coords, whole_graph)
        x = whole_graph['node_coords'][pose][0]
        y = whole_graph['node_coords'][pose][1]
        name = whole_graph['node_names'][pose]
        plt.text(x, y, f'{idx+1} - {pose}:{name}', color='brown', size=4)

    # Create a color map
    red_cmap = plt.get_cmap('Reds')
    colors = np.linspace(0, 1, len(trajectory[0]))
    red_colors = red_cmap(colors)
    selected_color = red_colors

    # Plot the points with color gradient
    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        plt.plot(x, y, color=selected_color[idx], marker='.', markersize=2, alpha=0.9)
    plt.savefig(image_file, dpi=300)

    idx_pairs = np.where(
        np.diff(
            np.hstack(([False], np.logical_not(agrees_with_oracle),
                       [False]))))[0].reshape(-1, 2)

    if len(idx_pairs) == 0:
        start_longest_disagreement = -1
    else:
        start_longest_disagreement = (
            idx_pairs[np.diff(idx_pairs, axis=1).argmax(), 0])

    if do_intervene:
        torch.save(planner.model.state_dict(), args.network_file)
        # Save the model with index every 10 maps
        if args.current_seed % 10 == 9:
            torch.save(
                planner.model.state_dict(),
                os.path.join(args.save_dir, 'logs', f'{args.current_seed}_lsp_int.pt'))
    else:
        return start_longest_disagreement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a figure for results from the interpretability project.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--current_seed', type=int, required=True)
    parser.add_argument('--data_file_base_name', type=str, required=True)
    parser.add_argument('--resolution', type=float, required=True)
    parser.add_argument('--cache_path', type=str, required=False)
    parser.add_argument('--network_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    args = parser.parse_args()
    # args.current_seed = 2
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)

    # Get the highest disagreed step to intervene
    args.intervene_at = intervention_pipeline(
        args, do_intervene=False)

    # reinstating seed is helping prevent the target object
    # from changing when intevention pipeline is called twice
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    print("Intervining at step: ", args.intervene_at)
    intervention_pipeline(args)

    # Create a log file for data completion
    open(os.path.join(
            args.save_dir,
            'data_completion_logs',
            f'{args.data_file_base_name}_{args.current_seed}.txt'), "x")
