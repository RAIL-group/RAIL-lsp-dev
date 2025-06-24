import numpy as np
import gridmap


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
