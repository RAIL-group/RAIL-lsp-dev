import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import procthor
import taskplan

IS_FCNN = True


def gen_data_main(args):
    coffee_objects = taskplan.utilities.utils.get_coffee_objects()
    # Load data for a given seed
    thor_data = procthor.procthor.ThorInterface(args=args, preprocess=coffee_objects)

    # Get the occupancy grid from data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph)

    # Iterate over the objects in whole graph and set
    # them as object to find to create training datas
    processed = set()
    for idx, target_obj in enumerate(whole_graph['obj_node_idx']):
        gen_name = whole_graph['node_names'][target_obj]
        if gen_name in processed:
            continue
        processed.add(gen_name)
        partial_map.target_obj = target_obj
        training_data = partial_map.get_training_data(fcnn=IS_FCNN)
        taskplan.utilities.utils.write_datum_to_file(args, training_data, idx, fcnn=IS_FCNN)

    plt.figure(figsize=(12, 4))
    plt.suptitle(f'Seed: [{args.current_seed}]')
    plt.subplot(131)
    img = whole_graph['graph_image']
    plt.imshow(img)
    plt.subplot(132)
    procthor.plotting.plot_graph_on_grid_old(grid, whole_graph)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)

    # plot top-down view from simulator
    plt.subplot(133)
    top_down_frame = thor_data.get_top_down_frame()
    plt.imshow(top_down_frame)
    plt.savefig(f'{args.save_dir}/data_completion_logs/{args.data_file_base_name}_{args.current_seed}.png', dpi=1000)


def get_args():
    parser = argparse.ArgumentParser(
        description="Data generation using ProcTHOR for Task Planning"
    )
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--data_file_base_name', type=str, required=False)
    parser.add_argument('--save_dir', type=str, required=False)
    parser.add_argument('--resolution', type=float, required=False)
    parser.add_argument('--cache_path', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    gen_data_main(args)
