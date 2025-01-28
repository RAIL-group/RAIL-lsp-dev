import copy
import numpy as np

import mr_task
from mrlsp.utils.utility import find_action_list_from_cost_matrix_using_lsa


class BaseMRTaskPlanner(object):
    def __init__(self, specification, verbose=True):
        self.specification = specification
        self.verbose = verbose
        self.dfa_planner = mr_task.DFAManager(specification)
        self.robot_poses = None
        self.container_nodes = None
        self.node_prop_dict = None

    def update(self, robot_poses, objects_found, container_nodes):
        self.robot_poses = robot_poses
        self.container_nodes = container_nodes

        # update the dfa planner
        self.dfa_planner.advance(objects_found)
        self.objects_to_find = self.dfa_planner.get_useful_props()

        # compute the node properties
        self._get_node_properties()

        if self.verbose:
            print('---------------------------')
            print(f'Task: {self.specification}')
            print(f'Objects to find: {self.objects_to_find}, DFA state: {self.dfa_planner.state}')

    def _get_node_properties(self):
        return


class OptimisticMRTaskPlanner(BaseMRTaskPlanner):
    def __init__(self, specification, verbose=True):
        super(OptimisticMRTaskPlanner, self).__init__(specification, verbose)

    def compute_joint_action(self):
        robot_nodes = [mr_task.core.RobotNode(mr_task.core.Node(
            location=(r_pose.x, r_pose.y))) for r_pose in self.robot_poses]
        # For optimistic planner, we assume that the objects are present in the containers.
        containers = [mr_task.core.Node(location=node.location, name=node.name, is_subgoal=False) for node in self.container_nodes]
        distances = mr_task.utils.get_inter_distances_nodes(
            containers, robot_nodes)
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
        joint_action = [mr_task.core.Action(node) for node in nodes]
        return joint_action, None
