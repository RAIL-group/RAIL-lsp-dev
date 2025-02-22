import copy
import numpy as np
import torch
import pouct_planner
import mr_task
from mr_task.core import Node
from .planner import BaseMRTaskPlanner
from mr_task.toy_environment import likelihoods
from mr_task.utils import get_inter_distances_nodes
from mr_task.learning.models import FCNN


class LearnedMRTaskPlanner(BaseMRTaskPlanner):
    def __init__(self, args, specification, device=None, verbose=True):
        super(LearnedMRTaskPlanner, self).__init__(args, specification, verbose)
        if device is not None:
            self.device = device
        else:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")

        self.node_prop_net = FCNN.get_net_eval_fn(args.network_file, device=self.device)

    def _update_node_properties(self):
        self.node_prop_dict = {}
        container_idxs = [node.name for node in self.unexplored_container_nodes]
        node_features_dict = mr_task.utils.prepare_fcnn_input(self.observed_graph, container_idxs, self.objects_to_find)
        object_container_prop_dict = {}
        for obj in self.objects_to_find:
            datum = {'node_feats': node_features_dict[obj]}
            object_container_prop_dict[obj] = self.node_prop_net(datum, container_idxs)

        for node in self.unexplored_container_nodes:
            for obj in self.objects_to_find:
                PS = object_container_prop_dict[obj][node.name]
                self.node_prop_dict[(node, obj)] = [PS, 0, 0]

    def compute_joint_action(self):
        if self.dfa_planner.has_reached_accepting_state():
            return [], None

        robot_nodes = [mr_task.core.RobotNode(Node(location=(r_pose.x, r_pose.y))) for r_pose in self.robot_poses]
        distances = get_inter_distances_nodes(self.explored_container_nodes + self.unexplored_container_nodes,
                                              robot_nodes)
        mrstate = mr_task.core.MRState(robots=robot_nodes,
                                       planner=copy.copy(self.dfa_planner),
                                       distances=distances,
                                       subgoal_prop_dict=self.node_prop_dict,
                                       known_space_nodes=self.explored_container_nodes,
                                       unknown_space_nodes=self.unexplored_container_nodes)
        def rollout_fn(state):
            return 0
        action, cost, [ordering, costs] = pouct_planner.core.po_mcts(
            mrstate, n_iterations=self.args.num_iterations, C=self.args.C, rollout_fn=rollout_fn)
        print("action ordering=", [(action.target_node.name, action.props) for action in ordering])
        print("costs=", costs)
        return ordering[:len(self.robot_poses)], cost
