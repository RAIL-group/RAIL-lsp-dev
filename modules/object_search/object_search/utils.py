import numpy as np
import gridmap
import argparse


def compute_cost_and_trajectory(grid, path):
    '''This function returns the path cost, robot trajectory
    given the occupancy grid and the container poses the
    robot explored during object search.
    '''
    total_cost = 0
    trajectory = None
    occ_grid = np.copy(grid)

    for pose in path:
        occ_grid[int(pose.x), int(pose.y)] = 0

    for idx, pose in enumerate(path[:-1]):
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            occ_grid,
            start=[pose.x, pose.y],
            use_soft_cost=True,
            only_return_cost_grid=False)
        next_pose = path[idx + 1]

        cost = cost_grid[int(next_pose.x), int(next_pose.y)]

        total_cost += cost
        _, robot_path = get_path([next_pose.x, next_pose.y],
                                 do_sparsify=False,
                                 do_flip=False)
        if trajectory is None:
            trajectory = robot_path
        else:
            trajectory = np.concatenate((trajectory, robot_path), axis=1)

    return total_cost, trajectory


def get_command_line_parser():
    parser = argparse.ArgumentParser(description='Object Search')
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--inflation_radius_m', type=float, default=0.0)
    parser.add_argument('--laser_max_range_m', type=float, default=10.0)
    parser.add_argument('--disable_known_grid_correction', action='store_true')
    parser.add_argument('--laser_scanner_num_points', type=int, default=1024)
    parser.add_argument('--field_of_view_deg', type=int, default=360)
    parser.add_argument('--step_size', type=float, default=1.8)
    parser.add_argument('--num_primitives', type=int, default=32)

    return parser
