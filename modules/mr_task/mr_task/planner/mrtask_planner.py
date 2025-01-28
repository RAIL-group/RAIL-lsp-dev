import copy
import numpy as np

import pouct_planner
import mr_task
from mr_task.core import Node
from .planner import BaseMRTaskPlanner
from mr_task.toy_environment import likelihoods
from mr_task.utils import get_inter_distances_nodes


class LearnedMRTaskPlanner(BaseMRTaskPlanner):
    def __init__(self, specification, verbose=True):
        super(LearnedMRTaskPlanner, self).__init__(specification, verbose)

    def _get_node_properties(self):
        self.node_prop_dict = {
            (node, obj): [likelihoods[node.name][obj], 0, 0]
            for node in self.container_nodes
            for obj in self.objects_to_find
        }

    def compute_joint_action(self):
        robot_nodes = [mr_task.core.RobotNode(Node(location=(r_pose.x, r_pose.y))) for r_pose in self.robot_poses]
        distances = get_inter_distances_nodes(self.container_nodes, robot_nodes)
        mrstate = mr_task.core.MRState(robots=robot_nodes,
                                       planner=copy.copy(self.dfa_planner),
                                       distances=distances,
                                       subgoal_prop_dict=self.node_prop_dict,
                                       known_space_nodes=[],
                                       unknown_space_nodes=self.container_nodes)
        action, cost = pouct_planner.core.get_best_joint_action(
            mrstate, num_robots=len(self.robot_poses), n_iterations=50000, C=100)
        return action, cost
