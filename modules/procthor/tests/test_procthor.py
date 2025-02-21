import numpy as np
import matplotlib.pyplot as plt
import procthor
from procthor import plotting
import pytest
from pathlib import Path


def get_args():
    args = lambda key: None
    args.save_dir = '/data/test_logs'
    args.current_seed = 0
    args.resolution = 0.05
    return args


@pytest.mark.timeout(15)
def test_thorinterface():
    '''This test plots occupancy grid and the original top-view image of the same ProcTHOR map.'''
    args = get_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / f'grid-scene-{args.current_seed}.png'

    thor_interface = procthor.ThorInterface(args=args)
    known_graph, known_grid, robot_pose, target_obj_info = thor_interface.gen_map_and_poses()

    assert len(known_graph.nodes) > 0
    assert len(known_graph.edges) > 0
    assert known_grid.size > 0
    assert len(robot_pose) == 2
    unique_vals = np.unique(known_grid)
    assert 1 in unique_vals and 0 in unique_vals
    assert 'name' in target_obj_info
    assert 'idx' in target_obj_info
    assert 'type' in target_obj_info
    assert 'container_idx' in target_obj_info

    top_down_image = thor_interface.get_top_down_image(orthographic=False)
    assert top_down_image.size > 0
    assert np.std(top_down_image) > 0
    assert np.mean(top_down_image) > 0

    plt.subplot(121)
    plotting.plot_graph_on_grid(known_grid, known_graph)
    x, y = robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)

    plt.subplot(122)
    plt.imshow(top_down_image)
    plt.axis('off')
    plt.savefig(save_file, dpi=600)


@pytest.mark.timeout(15)
def test_simulator():
    '''This test initializes the SceneGraphSimulator and performs a basic operational test.'''
    args = get_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / f'simulator-scene-{args.current_seed}.png'
    thor_interface = procthor.ThorInterface(args=args)
    known_graph, known_grid, robot_pose, target_obj_info = thor_interface.gen_map_and_poses()
    simulator = procthor.simulators.SceneGraphSimulator(known_graph, args, target_obj_info, known_grid, thor_interface)
    observed_graph, observed_grid, subgoals = simulator.initialize_graph_map_and_subgoals()

    assert len(observed_graph.nodes) > 0
    assert len(observed_graph.edges) > 0
    assert observed_grid.size > 0
    assert len(subgoals) > 0
    assert all([isinstance(s, int) for s in subgoals])
    assert all([s in observed_graph.container_indices for s in subgoals])
    assert len(observed_graph.object_indices) == 0
    top_down_image = simulator.get_top_down_image(orthographic=True)
    assert top_down_image.size > 0
    assert np.std(top_down_image) > 0
    assert np.mean(top_down_image) > 0

    plt.subplot(121)
    plotting.plot_graph_on_grid(observed_grid, observed_graph)
    x, y = robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)

    plt.subplot(122)
    plt.imshow(top_down_image)
    plt.axis('off')
    plt.savefig(save_file, dpi=600)
