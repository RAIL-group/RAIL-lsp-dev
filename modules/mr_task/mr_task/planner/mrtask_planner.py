import copy
import numpy as np

import pouct_planner
import mr_task
from mr_task.core import Node
from .planner import BaseMRTaskPlanner
from mr_task.toy_environment import likelihoods
from mr_task.utils import get_inter_distances_nodes


class LearnedMRTaskPlanner(BaseMRTaskPlanner):
    def __init__(self, args, specification, verbose=True):
        super(LearnedMRTaskPlanner, self).__init__(args, specification, verbose)

    # def _get_node_properties(self):
    #     self.node_prop_dict = {
    #         (node, obj): [likelihoods[node.name][obj], 0, 0]
    #         for node in self.container_nodes
    #         for obj in self.objects_to_find
    #     }

    def compute_joint_action(self):
        robot_nodes = [mr_task.core.RobotNode(Node(location=(r_pose.x, r_pose.y))) for r_pose in self.robot_poses]
        distances = get_inter_distances_nodes(self.container_nodes, robot_nodes)
        mrstate = mr_task.core.MRState(robots=robot_nodes,
                                       planner=copy.copy(self.dfa_planner),
                                       distances=distances,
                                       subgoal_prop_dict={},
                                       known_space_nodes=self.container_nodes,
                                       unknown_space_nodes=[])
        def rollout_fn(state):
            return 0
        action, cost, [ordering, costs] = pouct_planner.core.po_mcts(
            mrstate, n_iterations=self.args.num_iterations, C=self.args.C, rollout_fn=rollout_fn)
        print("action ordering=", [(action.target_node.name, action.props) for action in ordering])
        print("costs=", costs)
        return ordering[:len(self.robot_poses)], cost
