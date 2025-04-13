import numpy as np
import mr_task
from mr_task.core import Node
import gridmap
from common import compute_path_length

class MRTaskPlanningLoop(object):
    def __init__(self, robots, simulator, goal_reached_fn, verbose=True):
        self.robots = robots
        self.simulator = simulator
        self.graph, self.containers_idx = self.simulator.initialize_graph_and_containers()
        self.found_objects_name = ()
        self.joint_action = None
        self.counter = 0
        self.verbose = verbose
        self.ordering = {}
        self.goal_reached_fn = goal_reached_fn
        self.revealed_container_idxs = {}
        # following variables are required to know if robots are waiting
        self.prev_joint_action = None
        self.waiting_robots = []

    def __iter__(self):
        counter = 0
        while True:
            self.unexplored_container_nodes = [Node(is_subgoal=True,
                                                    name=idx,
                                                    location=self.simulator.known_graph.get_node_position_by_idx(idx))
                                               for idx in self.containers_idx]
            self.explored_container_nodes = [Node(name=idx, props=objects,
                                                  location=self.simulator.known_graph.get_node_position_by_idx(idx))
                                             for idx, objects in self.revealed_container_idxs.items()]

            yield {
                "robot_poses": [robot.pose for robot in self.robots],
                "explored_container_nodes": self.explored_container_nodes,
                "unexplored_container_nodes": self.unexplored_container_nodes,
                "object_found": self.found_objects_name,
                "observed_graph": self.graph,
                "observed_map": self.simulator.known_grid,
            }
            if self.verbose:
                print("PLANNING LOOP:")
                print(f"Step {counter}:")
                print(f'ROBOT STATUS:')
                for i, robot in enumerate(self.robots):
                    if self.joint_action is not None:
                        action_idx = self.joint_action[i].target_node.name
                        node_name = self.graph.get_node_name_by_idx(action_idx)
                        node_position = self.graph.get_node_position_by_idx(action_idx)
                        node_props = self.joint_action[i].props
                        text = f'R{i+1}: ({robot.pose.x}, {robot.pose.y}) ---> ({action_idx}:{node_name}:{node_position}->{node_props})'
                    else:
                        text = f'R{i+1}: ({robot.pose.x}, {robot.pose.y}) --> (None)'
                    print(text)
                print('EXPLORED CONTAINERS:')
                for idx, objects in self.revealed_container_idxs.items():
                    print(f'{idx}: {self.graph.get_node_name_by_idx(idx)} -> {objects}')
                print('UNEXPLORED CONTAINERS:')
                for idx in self.containers_idx:
                    print(f'{idx}: {self.graph.get_node_name_by_idx(idx)}', end=', ')
                print()

            if self.goal_reached_fn():
                break

            # Compute the trajectory from robot's pose to the target node for each robot
            paths = []
            distances = []
            cost_grids = []
            for i, robot in enumerate(self.robots):
                cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                    self.simulator.known_grid, [robot.pose.x, robot.pose.y], use_soft_cost=True)
                _, path = get_path(self.joint_action[i].target_node.location,
                                do_sparsify=False)
                cost_grids.append(cost_grid)
                paths.append(path)
                distances.append(compute_path_length(path))
            if self.prev_joint_action is not None:
                self.waiting_robots = [i for i, action in enumerate(self.joint_action)
                                       if action.target_node.name == self.prev_joint_action[i].target_node.name and
                                       (distances[i] - 2) <= 0 and action.target_node.name in self.revealed_container_idxs]
            non_waiting_action_dist_idx = [(distances[i], i) for i in range(len(distances)) if i not in self.waiting_robots]
            min_travel_distance, revealed_joint_action_idx = min(non_waiting_action_dist_idx)
            # min_travel_distance = np.min([distances[i] for i in range(len(distances)) if i not in self.waiting_robots])
            # print(f"Minimum travel distance: {min_travel_distance}")
            # min_travel_distance = np.min(distances)
            robots_path = [mr_task.utils.get_partial_path_upto_distance(cost_grid, path, min_travel_distance)
                           for cost_grid, path in zip(cost_grids, paths)]
            for robot, path in zip(self.robots, robots_path):
                robot.move(path)
            first_revealed_action = self.joint_action[revealed_joint_action_idx]
            first_revealed_action_idx = first_revealed_action.target_node.name
            # if the container is already revealed, we don't need to update the graph
            if not first_revealed_action_idx in self.revealed_container_idxs:
                self.graph, self.containers_idx = self.simulator.update_graph_and_containers(
                    observed_graph=self.graph,
                    containers=self.containers_idx,
                    chosen_container_idx=first_revealed_action_idx
                )
                self.found_objects_idx = self.graph.get_adjacent_nodes_idx(first_revealed_action_idx, filter_by_type=3)
                self.found_objects_name = tuple(self.graph.get_node_name_by_idx(idx) for idx in self.found_objects_idx)
                self.revealed_container_idxs[first_revealed_action_idx] = self.found_objects_name
                if len(self.waiting_robots) != 0:
                    waiting_action_idxs = [self.joint_action[i].target_node.name for i in self.waiting_robots]
                    for i in waiting_action_idxs:
                        self.found_objects_name += self.revealed_container_idxs[i]
            else:
                self.found_objects_name = self.revealed_container_idxs[first_revealed_action_idx]
                if len(self.waiting_robots) != 0:
                    waiting_action_idxs = [self.joint_action[i].target_node.name for i in self.waiting_robots]
                    for i in waiting_action_idxs:
                        self.found_objects_name += self.revealed_container_idxs[i]

            print(f"Revealed container {first_revealed_action_idx}:{self.graph.get_node_name_by_idx(first_revealed_action_idx)} with objects: {self.found_objects_name}")

            self.ordering[first_revealed_action_idx] = self.graph.get_node_name_by_idx(first_revealed_action_idx)
            counter += 1
            print("--------------------------------------------------")


    def update_joint_action(self, joint_action):
        if not self.goal_reached_fn():
            self.prev_joint_action = self.joint_action
            self.joint_action = joint_action
