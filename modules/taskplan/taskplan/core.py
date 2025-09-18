import math
import random
import numpy as np

import lsp
import gridmap
import taskplan
from taskplan.utilities.utils import get_action_costs


IS_FROM_LAST_CHOSEN_REWARD = 0 * 10.0


class Subgoal:
    def __init__(self, value) -> None:
        self.value = value
        self.props_set = False
        self.is_from_last_chosen = False
        self.is_obstructed = False
        self.prob_feasible = 1.0
        self.delta_success_cost = 0.0
        self.exploration_cost = 0.0
        self.negative_weighting = 0.0
        self.positive_weighting = 0.0

        self.counter = 0
        # Compute and cache the hash
        self.hash = hash(self.value)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def set_props(self,
                  prob_feasible,
                  is_obstructed=False,
                  delta_success_cost=0,
                  exploration_cost=0,
                  positive_weighting=0,
                  negative_weighting=0,
                  counter=0,
                  last_observed_pose=None,
                  did_set=True):
        self.props_set = did_set
        self.just_set = did_set
        self.prob_feasible = prob_feasible
        self.is_obstructed = is_obstructed
        self.delta_success_cost = delta_success_cost
        self.exploration_cost = exploration_cost
        self.positive_weighting = positive_weighting
        self.negative_weighting = negative_weighting
        self.counter = counter
        self.last_observed_pose = last_observed_pose


class PartialMap:
    ''' This class is responsible for creating the core capabilites like
    creating partial graph from the full graph, getting subgoals, performing
    change is graph based on action, etc.
    container_nodes carries the information about container poses
    '''
    def __init__(self, graph, grid=None, distinct=False):
        self.org_node_feats = graph['graph_nodes']
        self.org_edge_index = graph['graph_edge_index']
        self.org_node_names = graph['node_names']
        self.cnt_node_idx = graph['cnt_node_idx']
        self.obj_node_idx = graph['obj_node_idx']
        self.node_coords = graph['node_coords']
        self.idx_map = graph['idx_map']
        # self.distances = graph['distances']
        self.distinct = distinct  # when true looks for specific object instance

        self.target_obj = random.sample(self.obj_node_idx, 1)[0]
        self.container_poses = self._get_container_poses()

        if grid is not None:
            self.grid = grid

    def _get_container_poses(self):
        return {container_idx: tuple(self.node_coords[container_idx][0:2])
                for container_idx in self.cnt_node_idx}

    def _get_object_free_graph(self):
        # Trimdown all object nodes to get an object node free graph
        obj_count = len(self.obj_node_idx)

        temp = self.org_edge_index.copy()
        temp[0] = temp[0][:-obj_count:]
        temp[1] = temp[1][:-obj_count:]

        return {
            'node_feats': self.org_node_feats[:-obj_count:],
            'edge_index': temp
        }

    def initialize_graph_and_subgoals(self, seed=0):
        random.seed(seed)
        if self.distinct:
            target_obj_container_idx = [self.org_edge_index[0][self.org_edge_index[1].index(self.target_obj)]]
        else:
            # Find the container index containing the target object
            all_target = []
            for obj_idx in self.obj_node_idx:
                if self.org_node_names[obj_idx] == self.org_node_names[self.target_obj]:
                    all_target.append(obj_idx)
            target_obj_container_idx = {self.org_edge_index[0][self.org_edge_index[1].index(obj_idx)]
                                        for obj_idx in all_target}
        self.target_container = target_obj_container_idx

        # select 50% or above nodes as subgoals from the original
        # container nodes, but no less than 2
        cnt_count = len(self.cnt_node_idx)
        lb_sample = min(cnt_count, 2)
        num_of_val_to_choose = max(lb_sample, random.sample(list(range(
            cnt_count // 2, cnt_count)), 1)[0])
        subgoal_idx = random.sample(self.cnt_node_idx, num_of_val_to_choose)
        for target_obj_cnt in target_obj_container_idx:
            if target_obj_cnt not in subgoal_idx:
                subgoal_idx.append(target_obj_cnt)
        subgoal_idx = sorted(subgoal_idx)

        # Extract the container nodes' index that were not chosen
        # as subgoal nodes
        cnt_to_reveal_idx = [xx
                             for xx in self.cnt_node_idx
                             if xx not in subgoal_idx]

        # Check if the cnt_nodes to reveal had any connections in original
        # graph and update the initial graph adding those connection and nodes
        # they were connected to
        graph = self._get_object_free_graph().copy()

        for node_idx in cnt_to_reveal_idx:
            # if the node has connections in original
            # update graph: add edge to object node
            # append revealed object's feature to node_feats
            connected_obj_idx = [
                self.org_edge_index[1][idx]
                for idx, node in enumerate(self.org_edge_index[0])
                if node == node_idx
            ]

            for obj_idx in connected_obj_idx:
                graph['edge_index'][0].append(node_idx)
                graph['edge_index'][1].append(len(graph['node_feats']))
                graph['node_feats'].append(self.org_node_feats[obj_idx])

        return graph, subgoal_idx

    def update_graph_and_subgoals(self, subgoals, chosen_subgoal=None):
        if chosen_subgoal is not None:
            subgoals.remove(chosen_subgoal)

        subgoal_idx = subgoals
        cnt_to_reveal_idx = [xx
                             for xx in self.cnt_node_idx
                             if xx not in subgoal_idx]

        graph = self._get_object_free_graph()

        for node_idx in cnt_to_reveal_idx:
            # if the node has connections in original
            # update graph: add edge to object node
            # append revealed object's feature to node_feats
            connected_obj_idx = [
                self.org_edge_index[1][idx]
                for idx, node in enumerate(self.org_edge_index[0])
                if node == node_idx
            ]

            for obj_idx in connected_obj_idx:
                graph['edge_index'][0].append(node_idx)
                graph['edge_index'][1].append(len(graph['node_feats']))
                graph['node_feats'].append(self.org_node_feats[obj_idx])

        return graph, subgoal_idx

    def prepare_gcn_input(self, curr_graph, subgoals, simplify=False):
        # add the target node and connect it to all the subgoal nodes
        # with edges
        graph = curr_graph.copy()
        if simplify:
            graph = self._get_object_free_graph()

        len_nf = len(graph['node_feats'])
        for subgoal in subgoals:
            graph['edge_index'][0].append(subgoal)
            graph['edge_index'][1].append(len_nf)
            graph['node_feats'].append(self.org_node_feats[self.target_obj])
            len_nf += 1
        is_subgoal = [0] * len_nf
        is_target = [0] * len_nf
        is_target[-len(subgoals):] = [1] * len(subgoals)
        for subgoal_idx in subgoals:
            is_subgoal[subgoal_idx] = 1
        graph['is_subgoal'] = is_subgoal
        graph['is_target'] = is_target
        return graph

    def prepare_fcnn_input(self, subgoals):
        # there will be a data for all subgoals
        # data will be node features of the room, the subgooal and the target
        # the target will be the same for all subgoals
        # the subgoal will be different for each subgoal
        # the room will be the room of the subgoal
        # the label will be 1 if the subgoal is the target and 0 otherwise
        input = []
        for subgoal in subgoals:
            temp = []
            # find the room of the subgoal from edge_index of graph
            room = self.org_edge_index[0][self.org_edge_index[1].index(subgoal)]
            temp.extend(self.org_node_feats[room])
            temp.extend(self.org_node_feats[subgoal])
            temp.extend(self.org_node_feats[self.target_obj])

            # add to input
            input.append(temp)
        return {'node_feats': input}

    def get_training_data(self, simplify=False, fcnn=False):
        if fcnn:
            _, subgoals = self.initialize_graph_and_subgoals()
            input_graph = self.prepare_fcnn_input(subgoals)

            # create the label
            label = []
            for subgoal in subgoals:
                if subgoal in self.target_container:
                    label.append(1)
                else:
                    label.append(0)
            input_graph['labels'] = label
            return input_graph

        current_graph, subgoals = self.initialize_graph_and_subgoals()
        input_graph = self.prepare_gcn_input(current_graph, subgoals, simplify)

        label = [0] * len(input_graph['node_feats'])
        for target_container in self.target_container:
            label[target_container] = 1
        input_graph['labels'] = label

        return input_graph

    def set_room_info(self, robot_pose, rooms):
        self.room_info = {}
        robot_room = taskplan.utilities.utils.get_robots_room_coords(
            self.grid, robot_pose, rooms, return_idx=True)
        self.room_info[robot_pose] = robot_room

        for idx, room in enumerate(rooms):
            self.room_info[room['position']] = idx + 1

        for container in self.container_poses:
            container_room = self.org_edge_index[0][
                self.org_edge_index[1].index(container)]
            container_coords = self.container_poses[container]
            self.room_info[container_coords] = container_room


def get_top_n_frontiers_new(frontiers, n, robot_pose, partial_map):
    """This heuristic is for retrieving the 'best' N frontiers"""

    # we want to pick the N most likely from the starting room and then the
    # next M most likely absent those

    # need the information which frontier is in which room
    # also the robot is in which room
    # the robot room info for the initial_robot_pose cannot be calculated using partial map

    # first get the room of the robot
    robot_room = partial_map.room_info[robot_pose]

    # then get the (n-m) most likely containers from the other rooms

    frontiers = [f for f in frontiers if f.prob_feasible > 0]

    h_prob = {s: s.prob_feasible for s in frontiers}

    fs_prob = sorted(list(frontiers), key=lambda s: h_prob[s], reverse=True)

    seen = set()
    fs_collated = []

    # then get the n most likely containers from that room
    for front_p in fs_prob:
        front_coord = partial_map.container_poses[front_p.value]
        front_room = partial_map.room_info[front_coord]
        if front_p not in seen and front_room == robot_room:
            seen.add(front_p)
            fs_collated.append(front_p)

    for front_p in fs_prob:
        front_coord = partial_map.container_poses[front_p.value]
        front_room = partial_map.room_info[front_coord]
        if front_p not in seen:
            seen.add(front_p)
            fs_collated.append(front_p)

    assert len(fs_collated) == len(seen)
    assert len(fs_collated) == len(fs_prob)

    return fs_collated[0:n]


def get_best_expected_cost_and_frontier_list(
        subgoals, partial_map, robot_pose, destination, num_frontiers_max,
        alternate_sampling=False):
    for subgoal in subgoals:
        subgoal.delta_success_cost = get_action_costs()['pick']
    # Get robot distances
    robot_distances = get_robot_distances(
        partial_map.grid, robot_pose, subgoals)

    # Get goal distances
    if destination is None:
        goal_distances = {subgoal: robot_distances[subgoal]
                          for subgoal in subgoals}
    else:
        goal_distances = get_robot_distances(
            partial_map.grid, destination, subgoals)

    # Calculate top n subgoals
    if alternate_sampling:
        subgoals = get_top_n_frontiers_new(
            subgoals, num_frontiers_max, robot_pose, partial_map)
    else:
        subgoals = lsp.core.get_top_n_frontiers(
            subgoals, goal_distances, robot_distances, num_frontiers_max)

    # Get subgoal pair distances
    subgoal_distances = get_subgoal_distances(partial_map.grid, subgoals)

    distances = {
        'frontier': subgoal_distances,
        'robot': robot_distances,
        'goal': goal_distances,
    }

    out = lsp.core.get_lowest_cost_ordering(subgoals, distances)
    return out


def get_robot_distances(grid, robot_pose, subgoals):
    ''' This function returns distance from the robot to the subgoals
    where poses are stored in grid cell coordinates.'''
    robot_distances = dict()

    occ_grid = np.copy(grid)
    occ_grid[int(robot_pose[0])][int(robot_pose[1])] = 0

    for subgoal in subgoals:
        occ_grid[int(subgoal.pos[0]), int(subgoal.pos[1])] = 0

    cost_grid = gridmap.planning.compute_cost_grid_from_position(
        occ_grid,
        start=[
            robot_pose[0],
            robot_pose[1]
        ],
        use_soft_cost=True,
        only_return_cost_grid=True)

    # Compute the cost for each frontier
    for subgoal in subgoals:
        f_pt = subgoal.pos
        cost = cost_grid[int(f_pt[0]), int(f_pt[1])]

        if math.isinf(cost):
            cost = 100000000000
            subgoal.set_props(prob_feasible=0.0, is_obstructed=True)
            subgoal.just_set = False

        robot_distances[subgoal] = cost

    return robot_distances


def get_subgoal_distances(grid, subgoals):
    ''' This function returns distance from any subgoal to other subgoals
    where poses are stored in grid cell coordinates.'''
    subgoal_distances = {}
    occ_grid = np.copy(grid)
    for subgoal in subgoals:
        occ_grid[int(subgoal.pos[0]), int(subgoal.pos[1])] = 0
    for idx, sg_1 in enumerate(subgoals[:-1]):
        start = sg_1.pos
        cost_grid = gridmap.planning.compute_cost_grid_from_position(
            occ_grid,
            start=start,
            use_soft_cost=True,
            only_return_cost_grid=True)
        for sg_2 in subgoals[idx + 1:]:
            fsg_set = frozenset([sg_1, sg_2])
            fpoints = sg_2.pos
            cost = cost_grid[int(fpoints[0]), int(fpoints[1])]
            subgoal_distances[fsg_set] = cost

    return subgoal_distances


def compute_path_cost(grid, path):
    ''' This function returns the total path and path cost
    given the occupancy grid and the trjectory as poses, the
    robot has visited througout the object search process,
    where poses are stored in grid cell coordinates.'''
    total_cost = 0
    total_path = None
    occ_grid = np.copy(grid)

    for point in path:
        occ_grid[int(point[0]), int(point[1])] = 0

    for idx, point in enumerate(path[:-1]):
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            occ_grid,
            start=point,
            use_soft_cost=True,
            only_return_cost_grid=False)
        next_point = path[idx + 1]

        cost = cost_grid[int(next_point[0]), int(next_point[1])]

        total_cost += cost
        did_plan, robot_path = get_path([next_point[0], next_point[1]],
                                        do_sparsify=False,
                                        do_flip=False)
        if total_path is None:
            total_path = robot_path
        else:
            total_path = np.concatenate((total_path, robot_path), axis=1)

    return total_cost, total_path
