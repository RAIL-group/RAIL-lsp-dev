import random
import math
import numpy as np
import lsp
from gridmap.constants import UNOBSERVED_VAL
from gridmap import laser, mapping, utils


class SceneGraphSimulator:
    def __init__(self,
                 known_graph,
                 args,
                 target_obj_info,
                 known_grid=None,
                 thor_interface=None,
                 verbose=True):
        self.known_graph = known_graph
        self.args = args
        self.target_obj_info = target_obj_info
        self.known_grid = known_grid
        self.thor_interface = thor_interface
        self.verbose = verbose

    def get_top_down_image(self, orthographic=True):
        if self.thor_interface is None:
            raise ValueError("Thor Interface is not set")

        return self.thor_interface.get_top_down_image(orthographic=orthographic)

    def initialize_graph_grid_and_containers(self):
        random.seed(self.args.current_seed)
        # Select half of the containers as subgoals or at least two
        cnt_count = len(self.known_graph.container_indices)
        lb_sample = min(cnt_count, 2)
        num_of_val_to_choose = max(lb_sample, random.sample(list(range(
            cnt_count // 2, cnt_count)), 1)[0])
        unexplored_containers = set(random.sample(self.known_graph.container_indices, num_of_val_to_choose))
        target_obj_info = [self.target_obj_info] if isinstance(self.target_obj_info, dict) else self.target_obj_info
        for target_obj in target_obj_info:
            unexplored_containers.update(target_obj['container_idxs'])
        unexplored_containers = sorted(unexplored_containers)

        observed_graph = self.known_graph.get_object_free_graph()
        observed_grid = self.known_grid.copy()

        # Reveal container nodes not chosen as subgoals
        cnt_to_reveal_idx = [xx
                             for xx in self.known_graph.container_indices
                             if xx not in unexplored_containers]
        for node_idx in cnt_to_reveal_idx:
            connected_obj_idx = self.known_graph.get_adjacent_nodes_idx(node_idx, filter_by_type=3)
            for obj_idx in connected_obj_idx:
                o_idx = observed_graph.add_node(self.known_graph.nodes[obj_idx].copy())
                observed_graph.add_edge(node_idx, o_idx)

        return observed_graph, observed_grid, unexplored_containers

    def update_graph_grid_and_containers(self, observed_graph, observed_grid, containers, chosen_container_idx=None):
        if chosen_container_idx is None:
            return observed_graph, observed_grid, containers

        unexplored_containers = [s for s in containers if s != chosen_container_idx]
        observed_graph = observed_graph.copy()

        # Add objects from chosen container to the graph
        connected_obj_idx = self.known_graph.get_adjacent_nodes_idx(chosen_container_idx, filter_by_type=3)
        for obj_idx in connected_obj_idx:
            o_idx = observed_graph.add_node(self.known_graph.nodes[obj_idx].copy())
            observed_graph.add_edge(chosen_container_idx, o_idx)

        return observed_graph, observed_grid, unexplored_containers


class SceneGraphFrontierSimulator(lsp.simulators.Simulator):
    def __init__(self,
                 known_graph,
                 args,
                 target_obj_info,
                 known_grid=None,
                 thor_interface=None,
                 verbose=True):
        self.known_graph = known_graph
        self.args = args
        self.target_obj_info = target_obj_info
        self.known_grid = known_grid
        self.known_map = known_grid.copy()  # for compatibility with Simulator class
        self.thor_interface = thor_interface
        self.verbose = verbose
        self.resolution = args.resolution
        self.inflation_radius = args.inflation_radius_m / self.resolution
        self.frontier_grouping_inflation_radius = 0
        self.laser_max_range_m = args.laser_max_range_m
        self.disable_known_grid_correction = args.disable_known_grid_correction

        self.inflated_known_grid = utils.inflate_grid(
            self.known_grid, inflation_radius=self.inflation_radius)

        self.laser_scanner_num_points = args.laser_scanner_num_points
        self.directions = laser.get_laser_scanner_directions(
            num_points=self.laser_scanner_num_points,
            field_of_view_rad=math.radians(args.field_of_view_deg))
        self.unexplored_containers_known_graph = list(self.known_graph.container_indices)
        self.unexplored_containers_observed_graph = []
        self.observed_to_known_node_idx_map = {}

    def get_top_down_image(self, robot=None, orthographic=True):
        if self.thor_interface is None:
            raise ValueError("Thor Interface is not set")
        robot_pose = None if robot is None else (robot.pose.x, robot.pose.y, robot.pose.yaw)
        return self.thor_interface.get_top_down_image(robot_pose=robot_pose, orthographic=orthographic)

    def get_image(self, robot):
        robot_pose = robot.pose.x, robot.pose.y, robot.pose.yaw
        return self.thor_interface.get_egocentric_image(robot_pose=robot_pose)

    def update_graph_and_map(self, robot, observed_graph, observed_grid):
        ranges, observed_grid, visible_region = self.get_laser_scan_and_update_map(robot, observed_grid, True)
        visibility_mask = utils.inflate_grid(visible_region, 1.8, -0.1, 1.0)
        inflated_grid = self.get_inflated_grid(observed_grid, robot)
        inflated_grid = mapping.get_fully_connected_observed_grid(
            inflated_grid, robot.pose)
        frontiers = self.get_updated_frontier_set(inflated_grid, robot, set())

        observed_graph = observed_graph.copy()

        # Add containers to graph if they exist in known region of observed grid
        for node_idx in self.unexplored_containers_known_graph:
            if node_idx in self.observed_to_known_node_idx_map.values():
                continue
            x, y = self.known_graph.get_node_position_by_idx(node_idx)
            if observed_grid[x, y] != UNOBSERVED_VAL:
                cnt_idx = observed_graph.add_node(self.known_graph.nodes[node_idx].copy())
                room_idx = self.known_graph.get_parent_node_idx(node_idx)
                observed_graph.add_edge(room_idx, cnt_idx)
                self.unexplored_containers_observed_graph.append(cnt_idx)
                self.observed_to_known_node_idx_map[cnt_idx] = node_idx

        # If the robot goes close to a container, reveal the objects within it
        newly_found_objects = set()
        for node_idx in self.unexplored_containers_observed_graph:
            point = observed_graph.get_node_position_by_idx(node_idx)
            if self.is_robot_close_to_point(robot, point):
                known_idx = self.observed_to_known_node_idx_map[node_idx]
                self.unexplored_containers_observed_graph.remove(node_idx)
                self.unexplored_containers_known_graph.remove(known_idx)
                connected_obj_idx = self.known_graph.get_adjacent_nodes_idx(known_idx, filter_by_type=3)
                objects_found = [self.known_graph.get_node_name_by_idx(idx) for idx in connected_obj_idx]
                newly_found_objects.update(objects_found)
                for obj_idx in connected_obj_idx:
                    o_idx = observed_graph.add_node(self.known_graph.nodes[obj_idx].copy())
                    observed_graph.add_edge(node_idx, o_idx)
                    self.observed_to_known_node_idx_map[o_idx] = obj_idx

        containers = self.unexplored_containers_observed_graph.copy()

        return {
            'observed_graph': observed_graph,
            'observed_grid': observed_grid,
            'inflated_grid': inflated_grid,
            'containers': containers,
            'frontiers': frontiers,
            'ranges': ranges,
            'visibility_mask': visibility_mask,
            'newly_found_objects': newly_found_objects
        }

    def is_robot_close_to_point(self, robot, point):
        return (np.abs(robot.pose.x - point[0]) < 1 * self.args.step_size
                and np.abs(robot.pose.y - point[1]) < 1 * self.args.step_size)

    def get_updated_frontier_set(self, inflated_grid, robot, saved_frontiers):
        """Compute the frontiers, store the new ones and compute properties."""
        new_frontiers = lsp.core.get_frontiers(
            inflated_grid,
            group_inflation_radius=self.frontier_grouping_inflation_radius)
        saved_frontiers = lsp.core.update_frontier_set(saved_frontiers,
                                                       new_frontiers)
        return saved_frontiers
