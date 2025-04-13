import copy
import numpy as np

import mr_task
from mr_task.core import Node, RobotNode, Action
from mrlsp.utils.utility import find_action_list_from_cost_matrix_using_lsa


class BaseMRTaskPlanner(object):
    def __init__(self, args, specification, verbose=True):
        self.specification = specification
        self.args = args
        self.verbose = verbose
        self.dfa_planner = mr_task.DFAManager(specification)
        self.objects_to_find = self.dfa_planner.get_useful_props()
        self.observed_graph = None
        self.robot_poses = None
        self.explored_container_nodes = None
        self.unexplored_container_nodes = None
        self.node_prop_dict = None

    def update(self, observations, robot_poses, explored_container_nodes, unexplored_container_nodes, objects_found):
        self.observed_graph = observations['observed_graph']
        self.observed_map = observations['observed_map']
        self.robot_poses = robot_poses
        self.explored_container_nodes = explored_container_nodes
        self.unexplored_container_nodes = unexplored_container_nodes

        # update the dfa planner
        self.dfa_planner.advance(objects_found)
        self.objects_to_find = self.dfa_planner.get_useful_props()

        # compute the node properties
        self._update_node_properties()

        if self.verbose:
            print('---------------------------')
            print(f'Task: {self.specification}')
            print(f'Objects to find: {self.objects_to_find}, DFA state: {self.dfa_planner.state}')


    def _update_node_properties(self):
        return


class OptimisticMRTaskPlanner(BaseMRTaskPlanner):
    def __init__(self, args, specification, verbose=True):
        super(OptimisticMRTaskPlanner, self).__init__(args, specification, verbose)
        self.known_container_idx = []
        self.useful_known_containers = []
        self.useful_known_containers_props = []

    def update(self, observations, robot_poses, explored_container_nodes, unexplored_container_nodes, objects_found):
        super().update(
            observations, robot_poses, explored_container_nodes, unexplored_container_nodes, objects_found)
        for container in explored_container_nodes:
            if container.name not in self.known_container_idx:
                self.known_container_idx.append(container.name)
                # see if the container is useful in the future
                if set(self.objects_to_find) & set(objects_found):
                    self.useful_known_containers.append(container)
                    self.useful_known_containers_props.extend([o for o in objects_found])
                    if self.verbose:
                        print("Found useful container:", container.name, " location:", container.location, " objects", objects_found)


    def compute_joint_action(self):
        if self.dfa_planner.has_reached_accepting_state():
            return None, None

        robot_nodes = [RobotNode(Node(location=(r_pose.x, r_pose.y)))
                       for r_pose in self.robot_poses]
        # For optimistic planner, we assume that the objects are present in the containers.
        containers = [Node(location=node.location, name=node.name, is_subgoal=False)
                       for node in self.unexplored_container_nodes]
        # if terminal state can be reached by exploring explored containers, then just go to them.
        planner = copy.copy(self.dfa_planner)
        print("tuple of useful_known_containers_props", tuple(self.useful_known_containers_props))
        planner.advance(tuple(self.useful_known_containers_props))
        if planner.has_reached_accepting_state():
            containers = self.useful_known_containers

        distances = mr_task.utils.get_inter_distances_nodes(
            containers, robot_nodes, observed_map=self.observed_map)
        # To make sure that this function returns action object
        cost_dictionary = [None for _ in range(len(self.robot_poses))]
        for i in range(len(self.robot_poses)):
            cost_dictionary[i] = {
                container: distances[(robot_nodes[i].start, container)]
                for container in containers
            }
        node_matrix = np.array([list(cd.keys()) for cd in cost_dictionary])
        cost_matrix = np.array([list(cd.values()) for cd in cost_dictionary])
        # from the cost matrix, return the list of subgoals that has the least cost.
        nodes = find_action_list_from_cost_matrix_using_lsa(cost_matrix, node_matrix)
        joint_action = [Action(node) for node in nodes]
        return joint_action, None
