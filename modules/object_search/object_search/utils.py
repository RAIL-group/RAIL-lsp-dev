import numpy as np
import gridmap
from common import Pose
import lsp


def compute_cost_and_trajectory(grid, path, resolution=0.05, use_robot_model=True):
    '''This function returns the path cost, robot trajectory
    given the occupancy grid and the container poses the
    robot explored during object search.
    '''
    if use_robot_model:
        cost, trajectory = compute_cost_and_robot_trajectory(grid, path)
    else:
        cost, trajectory = compute_cost_and_dijkstra_trajectory(grid, path)

    return resolution * cost, trajectory


def compute_cost_and_dijkstra_trajectory(grid, path):
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


def compute_cost_and_robot_trajectory(grid, path):
    robot = lsp.robot.Turtlebot_Robot(pose=Pose(path[0].x, path[0].y, yaw=0))

    for _, next_pose in enumerate(path[1:]):
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            grid, start=[next_pose.x, next_pose.y],
            use_soft_cost=True,
            only_return_cost_grid=False
        )

        reached = False
        while not reached:
            _, robot_path = get_path([robot.pose.x, robot.pose.y],
                                     do_sparsify=True,
                                     do_flip=True)
            motion_primitives = robot.get_motion_primitives()
            costs, _ = lsp.primitive.get_motion_primitive_costs(grid,
                                                                cost_grid,
                                                                robot.pose,
                                                                robot_path,
                                                                motion_primitives,
                                                                do_use_path=True)
            robot.move(motion_primitives, np.argmin(costs))
            dist = Pose.cartesian_distance(robot.pose, next_pose)
            if dist < 1.0:
                reached = True

    trajectory = [[], []]
    for pose in robot.all_poses:
        trajectory[0].append(pose.x)
        trajectory[1].append(pose.y)

    return robot.net_motion, np.array(trajectory)
