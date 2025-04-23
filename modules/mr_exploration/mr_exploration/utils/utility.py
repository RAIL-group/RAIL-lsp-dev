from common import Pose
import random
import numpy as np
import math
import lsp
import mrlsp
import scipy
import common
import gridmap
from scipy.optimize import linear_sum_assignment
from gridmap.constants import (FREE_VAL,
                               UNOBSERVED_VAL,
                               COLLISION_VAL,
                               OBSTACLE_THRESHOLD)
import copy
from scipy.spatial import distance
import itertools


def get_path_middle_point(known_map, start, goal, args):
    """This function returns the middle point on the path from goal to the
    robot starting position"""
    inflation_radius = args.inflation_radius_m / args.base_resolution
    inflated_mask = gridmap.utils.inflate_grid(
        known_map, inflation_radius=inflation_radius)
    # Now sample the middle point
    cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
        inflated_mask, [goal.x, goal.y])
    _, path = get_path([start.x, start.y], do_sparsify=False, do_flip=False)
    row, col = path.shape
    x = path[0][col // 2]
    y = path[1][col // 2]
    new_start_pose = common.Pose(x=x, y=y, yaw=2 * np.pi * np.random.rand())
    return new_start_pose


# generate different start and goal poses for num_robots
def generate_start_and_goal(num_robots=1,
                            known_map=None,
                            same_start=False,
                            same_goal=False,
                            def_start=None,
                            def_goal=None):
    if (not same_start or not same_goal):

        start = [0 for i in range(num_robots)]
        goal = [0 for i in range(num_robots)]
        start_pose = [0 for i in range(num_robots)]
        goal_pose = [0 for i in range(num_robots)]
        for i in range(num_robots):
            while True:
                (x, y) = random.randint(0,
                                        len(known_map) - 1), random.randint(
                                            0,
                                            len(known_map[0]) - 1)
                if (known_map[x, y] == 0):
                    a = np.array([x, y])
                    start[i] = a
                    break

        for i in range(num_robots):
            while True:
                (x, y) = random.randint(0,
                                        len(known_map) - 1), random.randint(
                                            0,
                                            len(known_map[0]) - 1)
                if (known_map[x, y] == 0):
                    a = np.array([x, y])
                    goal[i] = a
                    break

        for i in range(num_robots):
            start_pose[i] = Pose(x=start[i][0],
                                 y=start[i][1],
                                 yaw=2 * math.pi * random.random())

            goal_pose[i] = Pose(x=goal[i][0],
                                y=goal[i][1],
                                yaw=2 * math.pi * random.random())
        print("not same start or goal")

    if same_start:
        print("same start")
        start_pose = [def_start for _ in range(num_robots)]

    if same_goal:
        print("same goal")
        goal_pose = [def_goal for _ in range(num_robots)]

    return start_pose, goal_pose


# To update the current grid with the current observation
def update_grid(robot_grid, observed_grid):
    previous = robot_grid
    now = observed_grid
    current = lsp.constants.UNOBSERVED_VAL * np.ones_like(previous)

    for idx, _ in np.ndenumerate(current):
        # more like and gate
        if previous[idx] == UNOBSERVED_VAL and now[idx] == UNOBSERVED_VAL:
            current[idx] = UNOBSERVED_VAL
        elif previous[idx] == COLLISION_VAL and now[idx] == COLLISION_VAL:
            current[idx] = COLLISION_VAL
        elif previous[idx] == COLLISION_VAL and now[idx] == UNOBSERVED_VAL:
            current[idx] = COLLISION_VAL
        elif previous[idx] == UNOBSERVED_VAL and now[idx] == COLLISION_VAL:
            current[idx] = COLLISION_VAL
        else:
            current[idx] = FREE_VAL
    return current


def get_fully_connected_global_grid_multirobot(occupancy_grid, poses):
    """Returns a version of the observed grid in which components
    unconnected to the region containing the robot are set to 'unobserved'.
    This useful for preventing the system from generating any frontiers
    that cannot be planned to. Also, certain geometrys may cause frontiers
    to be erroneously ruled out if "stray" observed space exists.
    """
    # Group the frontier points into connected components
    labels, _ = scipy.ndimage.label(
        np.logical_and(occupancy_grid < OBSTACLE_THRESHOLD,
                       occupancy_grid >= FREE_VAL))

    occupancy_grid = occupancy_grid.copy()
    robot_labels = [labels[int(pose.x), int(pose.y)] for pose in poses]
    robot_labels_mask = np.isin(labels, robot_labels)
    mask = np.logical_and(labels > 0, ~robot_labels_mask)
    occupancy_grid[mask] = UNOBSERVED_VAL

    return occupancy_grid


def get_robot_team(num_robots, start_poses, primitive_length, num_primitives, map_data):
    '''A function that returns robot team'''
    robot_team = []
    for i in range(num_robots):
        robot_team.append(mrlsp.robot.MRobot(
            i, num_robots, start_poses[i], primitive_length, num_primitives, map_data))

    return robot_team


def get_robot_subgoal_distances(global_inflated_grid, robot_poses, subgoals):
    subgoal_distances = []
    for pose in robot_poses:
        subgoal_distances.append(lsp.core.get_robot_distances(global_inflated_grid, pose, frontiers=subgoals))
    return subgoal_distances


def get_top_n_frontiers_multirobot(num_robots, frontiers, distances, n):
    goal_dist = distances['goal']
    all_robot_best_frontiers = []
    h_prob = {s: s.prob_feasible for s in frontiers}
    fs_prob = sorted(list(frontiers),
                        key=lambda s: h_prob[s],
                        reverse=True)
    # Select top 2 frontiers according to probability
    all_robot_best_frontiers = fs_prob[:2]
    best_frontiers = []
    for i in range(num_robots):
        robot_dist = distances['robot'][i]
        bf = lsp.core.get_top_n_frontiers(frontiers, goal_dist, robot_dist, n)
        if len(bf) == 0:
            print("No frontiers returned by get_top_n_frontiers")
            fs_prob = sorted(list(frontiers),
                             key=lambda s: h_prob[s],
                             reverse=True)
            bf = fs_prob[:n]
        best_frontiers.append(bf)

    for i in range(n):
        for bf in best_frontiers:
            if i < len(bf) and bf[i] not in all_robot_best_frontiers:
                all_robot_best_frontiers.append(bf[i])

    top_frontiers = all_robot_best_frontiers[:n]
    return top_frontiers


def get_multirobot_distances(robot_grid, robots, subgoals):
    subgoals = {copy.copy(s) for s in subgoals}
    distances = {}
    #distances['goal'] = lsp.core.get_goal_distances(robot_grid, goal_pose[0], frontiers=subgoals)
    distances['frontier'] = lsp.core.get_frontier_distances(robot_grid, frontiers=subgoals)
    distances['robot'] = get_robot_subgoal_distances(robot_grid, [robot.pose for robot in robots], subgoals)
    return distances


def find_action_list_from_cost_matrix_using_lsa(cost_matrix, subgoal_matrix):
    cost = cost_matrix
    num_robots = len(cost_matrix)
    left_robot = num_robots
    assigned_robot = 0
    joint_action = [None for i in range(num_robots)]
    count = 0
    while (left_robot != 0 and count < num_robots + 1):
        # find the lowest cost for the first 'k' robots, where k is the number of subgoals
        n_rows, n_cols = linear_sum_assignment(cost)
        for i, row in enumerate(n_rows):
            # assign the action to the robot if it is not previously assigned, i.e., not None
            if joint_action[row] is None:
                joint_action[row] = subgoal_matrix[row][n_cols[i]]
                assigned_robot += 1
                # replace the cost by a 'high number' so that it it doesn't get selected when doing lsa
                cost[row] = 1e11
            # decrement the left robot so that it loops and assigns to the remaining robot.
        left_robot = num_robots - assigned_robot
        count += 1
    # for every none items in the joint action, randomly assign a subgoal in the joint action that's not none
    if None in joint_action:
        non_none_items = [item for item in joint_action if item is not None]
        none_idx = [idx for idx, val in enumerate(joint_action) if val is None]
        for idx in none_idx:
            joint_action[idx] = np.random.choice(non_none_items)
    return joint_action



def robot_team_communicate_data(robots_within_range, robots, robot_grids):
    '''A function that updates every robot's observation in team_poses, team_occupancy_grids, team_subgoals'''

    # Update every robot's observation in local_occupancy_grid
    for i in range(len(robots)):
        robots[i].local_occupancy_grid = robot_grids[i]

    # Update every robot's occupancy grid
    for robot_team in robots_within_range:
        team_grid = lsp.constants.UNOBSERVED_VAL * np.ones_like(robots[0].local_occupancy_grid)
        for robot_tag in robot_team:
            team_grid = update_grid(team_grid, robots[robot_tag].local_occupancy_grid)
        # update every MRobot's global_occupancy_grid
        for robot_tag in robot_team:
            robots[robot_tag].global_occupancy_grid = team_grid


def find_robots_within_range(robots, communication_range):
    '''A function that takes input robot poses as list and communication range and outputs the set of robot teams
        that are within communication range of each other.'''
    robot_poses = np.array([[robot.pose.x, robot.pose.y] for robot in robots])
    robot_teams = set()
    distance_array = distance.cdist(robot_poses, robot_poses, 'euclidean')
    for i in range(len(robot_poses)):
        robot_teams.add(tuple(np.where(distance_array[i] <= communication_range)[0]))
    return robot_teams


def get_subgoal_areas(subgoals, global_occupancy_grid, known_map, inflation_radius):
    # Find the connected unexplored regions that are free space
    unobserved_free_space = np.logical_and(global_occupancy_grid == UNOBSERVED_VAL, known_map == FREE_VAL)
    labels, num_labels = scipy.ndimage.label(unobserved_free_space)

    # Create a binary mask for subgoal centroids
    subgoal_mask = np.zeros_like(global_occupancy_grid, dtype=np.uint8)
    for subgoal in subgoals:
        x, y = map(int, subgoal.centroid[:2])
        subgoal_mask[x, y] = 1

    # Inflate subgoal locations to help find touching regions
    expanded_frontiers = scipy.ndimage.maximum_filter(subgoal_mask, size=2*inflation_radius)

    # Get labels that intersect with the expanded frontier regions
    explorable_labels = np.unique(labels[(expanded_frontiers == 1) & (labels >= 0)])

    # Measure the area (number of free cells) for each explorable region
    explorable_regions = {}
    for label_id in explorable_labels:
        region_size = np.sum(labels == label_id)
        explorable_regions[label_id] = region_size

    # Link each subgoal to its corresponding region label (if valid)
    subgoal_to_label = {}
    for subgoal in subgoals:
        x, y = map(int, subgoal.centroid[:2])
        label_id = labels[x, y]
        if label_id in explorable_regions:
            subgoal_to_label[subgoal] = label_id
        else:
            subgoal_to_label[subgoal] = None  # Assign None if subgoal doesn't match any explorable region

    return explorable_regions, labels, subgoal_to_label
