import copy
import torch
from common import Pose
import taskplan
from taskplan.core import Subgoal
from taskplan.learning.models.gnn import Gnn
from taskplan.utilities import utils

NUM_MAX_FRONTIERS = 8


class Planner():
    def __init__(self, target_obj_info, args, device=None, verbose=True):
        self.args = args
        self.target_obj_info = target_obj_info
        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        self.verbose = verbose
        # self.partial_map = partial_map

    def update(self, graph, grid, subgoals, robot_pose):
        self.graph = graph
        self.grid = grid
        self.robot_pose = robot_pose
        self.new_subgoals = [s for s in subgoals]
        # Convert into `Subgoal class' in an attempt to match cost calculation
        # API's input format
        self.subgoals = []
        for idx in subgoals:
            pose = self.graph.get_node_position_by_idx(idx)[:2]
            # pose = Pose(pose[0], pose[1])
            self.subgoals.append(Subgoal(idx, pose))
        self._update_subgoal_properties()

    def _update_subgoal_properties():
        raise NotImplementedError


class KnownPlanner(Planner):
    ''' This planner has access to the target object location and can
    find the object in 1 step
    '''
    def __init__(self, target_obj_info, args, known_graph, known_grid, device=None, verbose=False, destination=None):
        super(KnownPlanner, self).__init__(
            target_obj_info, args, device, verbose)
        self.known_graph = known_graph
        self.known_grid = known_grid
        self.destination = destination

    def _update_subgoal_properties(self):
        for subgoal in self.subgoals:
            contained_obj_idx = self.known_graph.get_adjacent_nodes_idx(subgoal.id, filter_by_type=3)
            contained_obj_names = [self.known_graph.get_node_name_by_idx(idx) for idx in contained_obj_idx]
            if self.target_obj_info['name'] in contained_obj_names:
                subgoal.set_props(prob_feasible=1.0)
            else:
                subgoal.set_props(prob_feasible=0.0)
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:6.4f}|'
                    f'at= {self.graph.get_node_name_by_idx(subgoal.id)}'
                )

        if self.verbose:
            print(" ")

    def compute_selected_subgoal(self):
        min_cost, frontier_ordering = (
            taskplan.core.get_best_expected_cost_and_frontier_list(
                self.subgoals,
                self.known_grid,
                self.robot_pose,
                self.destination,
                num_frontiers_max=NUM_MAX_FRONTIERS))
        return frontier_ordering[0]


class ClosestActionPlanner(Planner):
    ''' This planner naively looks in the nearest container to find any object
    '''
    def __init__(self, target_obj_info, args, device=None, verbose=False,
                 destination=None):
        super(ClosestActionPlanner, self).__init__(
            target_obj_info, args, device, verbose)
        self.destination = destination

    def _update_subgoal_properties(self):
        for subgoal in self.subgoals:
            subgoal.set_props(prob_feasible=1.0)
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:6.4f}|'
                    f'at= {self.graph.get_node_name_by_idx(subgoal.id)}'
                )

        if self.verbose:
            print(" ")

    def compute_selected_subgoal(self):
        min_cost, frontier_ordering = (
            taskplan.core.get_best_expected_cost_and_frontier_list(
                self.subgoals,
                self.grid,
                self.robot_pose,
                self.destination,
                num_frontiers_max=NUM_MAX_FRONTIERS))
        return frontier_ordering[0]


class LearnedPlanner(Planner):
    ''' This planner calculates subgoal properties using the learned network
    and then uses LSP approach to pick the best available action (subgoal).
    '''
    def __init__(self, target_obj_info, args, device=None, verbose=True,
                 destination=None):
        super(LearnedPlanner, self).__init__(
            target_obj_info, args, device, verbose)
        self.destination = destination
        self.subgoal_property_net = Gnn.get_net_eval_fn(
            args.network_file, device=self.device)

    def _update_subgoal_properties(self):
        gcn_graph_input = utils.prepare_gcn_input(
            graph=self.graph,
            subgoals=self.new_subgoals,
            target_obj_info=self.target_obj_info,
        )
        prob_feasible_dict = self.subgoal_property_net(
            datum=gcn_graph_input,
            subgoals=self.subgoals
        )
        for subgoal in self.subgoals:
            subgoal.set_props(
                prob_feasible=prob_feasible_dict[subgoal])
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:6.4f}|'
                    f'at={self.graph.get_node_name_by_idx(subgoal.id)}'
                )

        if self.verbose:
            print(" ")

    def compute_selected_subgoal(self, return_cost=False):
        min_cost, frontier_ordering = (
            taskplan.core.get_best_expected_cost_and_frontier_list(
                self.subgoals,
                self.grid,
                self.robot_pose,
                self.destination,
                num_frontiers_max=NUM_MAX_FRONTIERS))
        if return_cost:
            return min_cost, frontier_ordering[0]
        return frontier_ordering[0]
