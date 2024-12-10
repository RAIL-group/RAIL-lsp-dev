import os
import random
import numpy as np
import matplotlib.pyplot as plt
import procthor
import pytest

import taskplan


def get_args():
    create_dir()
    args = lambda key: None
    args.current_seed = 0
    args.resolution = 0.05
    return args


def create_dir():
    main_path = '/data'
    sub_dir = 'test_logs'
    sub_path = os.path.join(main_path, sub_dir)
    if not os.path.exists(sub_path):
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        os.makedirs(sub_path)


@pytest.mark.timeout(15)
def test_reachable_grid():
    '''This test plots occupancy grid and the original top-view image of the same ProcTHOR map.'''
    args = get_args()
    save_file = f'/data/test_logs/grid-scene-{args.current_seed}.png'

    # # Get data for a send and extract initial object states
    thor_data = procthor.ThorInterface(args=args, preprocess=True)

    # Get the occupancy grid from thor_data
    grid = thor_data.occupancy_grid
    plt.subplot(121)
    img = np.transpose(grid)
    plt.imshow(img)

    top_down_frame = thor_data.get_top_down_frame()
    plt.subplot(122)
    plt.imshow(top_down_frame)
    plt.savefig(save_file, dpi=1200)

    assert grid.size > 0
    assert top_down_frame.size > 0
    unique_vals = np.unique(grid)
    assert 1 in unique_vals and 0 in unique_vals
    assert np.std(top_down_frame) > 0


@pytest.mark.timeout(15)
def test_graph_on_grid():
    '''This test plots occupancy grid and the original top-view image of the same ProcTHOR map.'''
    args = get_args()
    save_file = f'/data/test_logs/graph-on-grid-{args.current_seed}.png'

    # Get thor data for a send and extract initial object states
    thor_data = procthor.ThorInterface(args=args)
    # Get the occupancy grid from thor_data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from ProcTHOR data
    whole_graph = thor_data.get_graph(include_node_embeddings=False)
    assert len(whole_graph['nodes']) > 0
    assert len(whole_graph['edge_index']) > 0

    plt.subplot(121)
    procthor.plotting.plot_graph_on_grid(grid, whole_graph)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)

    # plot top-down view from simulator
    plt.subplot(122)
    top_down_frame = thor_data.get_top_down_frame()
    plt.imshow(top_down_frame)

    plt.savefig(save_file, dpi=1200)


def test_custom_object_graph():
    '''This test checks if the custom object graph is created correctly.'''
    args = get_args()
    args.current_seed = 7060

    obj_loc_dict = {
        'waterbottle': ['fridge', 'diningtable'],
        'coffeegrinds': ['diningtable', 'countertop', 'shelvingunit']
    }

    thor_data = procthor.ThorInterface(args=args, preprocess=obj_loc_dict)

    # Get the whole graph from ProcTHOR data
    whole_graph = thor_data.get_graph()
    assert 'waterbottle' in whole_graph['node_names'].values()
    assert 'coffeegrinds' in whole_graph['node_names'].values()

    plt.imshow(whole_graph['graph_image'])
    plt.savefig(f'/data/test_logs/custom-graph-{args.current_seed}.png', dpi=500)


def test_align_plot():
    args = get_args()

    # Load data for a given seed
    thor_data = procthor.ThorInterface(args=args)

    whole_graph = thor_data.get_graph()
    grid = thor_data.occupancy_grid

    init_robot_pose = thor_data.get_robot_pose()
    robot_poses = [init_robot_pose]
    robot_poses += random.sample(list(whole_graph['node_coords'].values()), 3)

    distance, trajectory = taskplan.core.compute_path_cost(grid, robot_poses)

    plt.clf()
    plt.figure(figsize=(12, 6))

    viridis_cmap = plt.get_cmap('viridis')

    colors = np.linspace(0, 1, len(trajectory[0]))
    line_colors = viridis_cmap(colors)

    plt.subplot(121)
    # 1 plot the top-down-view
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

    plt.subplot(122)
    # 2 plot the trajectory overlaied grid
    plotting_grid = procthor.plotting.make_plotting_grid(np.transpose(grid))
    plt.imshow(plotting_grid)
    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title('Path overlaied occupancy grid', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    image_filename = f'/data/test_logs/align_plot_{args.current_seed}.png'
    plt.savefig(image_filename, dpi=400)
    assert os.path.exists(image_filename)
