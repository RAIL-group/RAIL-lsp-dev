import os
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pddlstream.algorithms.search import solve_from_pddl

import procthor
import taskplan


def evaluate_main(args):
    # get the custom coffee related objects
    if 'coffee' in args.goal_type:
        coffee_objects = taskplan.utilities.utils.get_coffee_objects()
        preprocess = coffee_objects
    else:
        preprocess = True
    # Load data for a given seed
    thor_data = procthor.ThorInterface(args=args, preprocess=preprocess)

    # Get the occupancy grid from data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph, grid)
    partial_map.set_room_info(init_robot_pose, thor_data.rooms)

    if args.cost_type == 'learned':
        learned_data = {
            'partial_map': partial_map,
            'learned_net': args.network_file
        }
    else:
        learned_data = None
    # Instantiate PDDL for this map
    pddl = taskplan.pddl.helper.get_pddl_instance(
        whole_graph=whole_graph,
        map_data=thor_data,
        args=args,
        learned_data=learned_data
    )
    taskplan.utilities.utils.check_pddl_validity(pddl, args)

    cost_str = taskplan.utilities.utils.get_cost_string(args)

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'],
                                 planner=pddl['planner'], max_planner_time=120)

    taskplan.utilities.utils.check_plan_validity(plan, args)

    # Intialize logfile
    logfile = os.path.join(args.save_dir, args.logfile_name)
    with open(logfile, "a+") as f:
        f.write(f"LOG: {args.current_seed}\n")

    executed_actions, robot_poses, action_cost = taskplan.planners.task_loop.run(
        plan, pddl, partial_map, init_robot_pose, args)

    distance, trajectory = taskplan.core.compute_path_cost(partial_map.grid, robot_poses)
    distance += action_cost
    print(f"Planning cost: {distance}")

    with open(logfile, "a+") as f:
        # err_str = '' if did_succeed else '[ERR]'
        f.write(f"[Learn] s: {args.current_seed:4d}"
                f" | {cost_str}: {distance:0.3f}\n")

    plt.clf()
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"{pddl['goal']} - seed: [{args.current_seed}]", fontsize=6)

    plt.subplot(231)
    # 0 plot the plan
    taskplan.plotting.plot_plan(plan=executed_actions)

    plt.subplot(232)
    # 1 plot the whole graph
    plt.title('Whole scene graph', fontsize=6)
    graph_image = whole_graph['graph_image']
    plt.imshow(graph_image)
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])

    viridis_cmap = plt.get_cmap('viridis')

    colors = np.linspace(0, 1, len(trajectory[0]))
    line_colors = viridis_cmap(colors)

    plt.subplot(233)
    # 2 plot the top-dwon-view
    top_down_frame = thor_data.get_top_down_frame()
    offset = thor_data.plot_offset
    extent = thor_data.plot_extent
    plt.imshow(top_down_frame, extent=extent)
    plt.title('Top-down view of the map', fontsize=6)
    for idx, x in enumerate(trajectory[0]):
        x = x + offset[0]
        y = trajectory[1][idx] + offset[1]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.subplot(234)
    # 3 plot the graph overlaied image
    procthor.plotting.plot_graph_on_grid(grid, whole_graph)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title('Graph overlaied occupancy grid', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.subplot(235)
    # 4 plot the graph overlaied image
    plotting_grid = procthor.plotting.make_plotting_grid(np.transpose(grid))
    plt.imshow(plotting_grid)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title('Path overlaied occupancy grid', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)

    plt.subplot(236)
    # 5 plot the grid with trajectory viridis color
    plotting_grid = procthor.plotting.make_blank_grid(
        np.transpose(grid)
    )
    plt.imshow(plotting_grid)
    plt.title(f"{cost_str} Cost: {distance:0.3f}", fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)

    # Hide box and ticks
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=1000)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for Task Planning under uncertainty"
    )
    parser.add_argument('--current_seed', type=int, required=True)
    parser.add_argument('--image_filename', type=str, required=True)
    parser.add_argument('--logfile_name', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--resolution', type=float, required=True)
    parser.add_argument('--network_file', type=str, required=False)
    parser.add_argument('--goal_type', type=str, required=False)
    parser.add_argument('--cost_type', type=str, required=False)
    parser.add_argument('--cache_path', type=str, required=False)
    parser.add_argument('--fail_log', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # skip experiment if seed in ignore list
    taskplan.utilities.utils.check_skip_protocol(args)
    if args.cost_type != 'known':
        c_str = taskplan.utilities.utils.get_cost_string(args)
        file_name = f"fail_{c_str}.txt"
        args.fail_log = os.path.join(args.save_dir, file_name)
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    evaluate_main(args)
