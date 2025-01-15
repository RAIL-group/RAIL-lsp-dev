import copy
import torch

import taskplan
from taskplan.core import Subgoal
from taskplan.learning.models.gnn import Gnn
from taskplan.learning.models.fcnn import Fcnn


NUM_MAX_FRONTIERS = 8


class Planner():
    def __init__(self, args, partial_map=None, device=None, verbose=True):
        self.args = args
        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        self.verbose = verbose
        self.partial_map = partial_map

    def update(self, graph, subgoals, robot_pose):
        self.graph = graph
        self.robot_pose = robot_pose
        self.new_subgoals = [s for s in subgoals]
        # Convert into `Subgoal class' in an attempt to match cost calculation
        # API's input format
        self.subgoals = [copy.copy(Subgoal(value)) for value in self.new_subgoals]
        for subgoal in self.subgoals:
            subgoal.pos = self.partial_map.container_poses[subgoal]
        self._update_subgoal_properties()

    def _update_subgoal_properties():
        raise NotImplementedError


class KnownPlanner(Planner):
    ''' This planner has access to the target object location and can
    find the object in 1 step
    '''
    def __init__(self, args, partial_map, device=None, verbose=False,
                 destination=None):
        super(KnownPlanner, self).__init__(
            args, partial_map, device, verbose)
        self.destination = destination

    def _update_subgoal_properties(self):
        for subgoal in self.subgoals:
            if subgoal.value in self.partial_map.target_container:
                subgoal.set_props(prob_feasible=1.0)
            else:
                subgoal.set_props(prob_feasible=0.0)
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:6.4f}|'
                    f'at= {self.partial_map.org_node_names[subgoal.value]}'
                )

        if self.verbose:
            print(" ")

    def compute_selected_subgoal(self):
        min_cost, frontier_ordering = (
            taskplan.core.get_best_expected_cost_and_frontier_list(
                self.subgoals,
                self.partial_map,
                self.robot_pose,
                self.destination,
                num_frontiers_max=NUM_MAX_FRONTIERS,
                alternate_sampling=True))
        return frontier_ordering[0]


class ClosestActionPlanner(Planner):
    ''' This planner naively looks in the nearest container to find any object
    '''
    def __init__(self, args, partial_map, device=None, verbose=False,
                 destination=None):
        super(ClosestActionPlanner, self).__init__(
            args, partial_map, device, verbose)
        self.destination = destination

    def _update_subgoal_properties(self):
        for subgoal in self.subgoals:
            subgoal.set_props(prob_feasible=1.0)
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:6.4f}|'
                    f'at= {self.partial_map.org_node_names[subgoal.value]}'
                )

        if self.verbose:
            print(" ")

    def compute_selected_subgoal(self):
        min_cost, frontier_ordering = (
            taskplan.core.get_best_expected_cost_and_frontier_list(
                self.subgoals,
                self.partial_map,
                self.robot_pose,
                self.destination,
                num_frontiers_max=NUM_MAX_FRONTIERS,
                alternate_sampling=True))
        return frontier_ordering[0]


class LearnedPlanner(Planner):
    ''' This planner calculates subgoal properties using the learned network
    and then uses LSP approach to pick the best available action (subgoal).
    '''
    def __init__(self, args, partial_map, device=None, verbose=True,
                 destination=None, normalize=True, fcnn=True, assisstance=True):
        super(LearnedPlanner, self).__init__(
            args, partial_map, device, verbose)
        self.destination = destination
        self.is_fcnn = fcnn
        if fcnn:
            self.subgoal_property_net = Fcnn.get_net_eval_fn(
                args.network_file, device=self.device)
        else:
            self.subgoal_property_net = Gnn.get_net_eval_fn(
                args.network_file, device=self.device)
        self.normalize = normalize
        self.assisstance = assisstance

    def _update_subgoal_properties(self):
        if self.is_fcnn:
            self.gcn_graph_input = self.partial_map.prepare_fcnn_input(
                subgoals=self.new_subgoals)
        else:
            self.gcn_graph_input = self.partial_map.prepare_gcn_input(
                curr_graph=self.graph,
                subgoals=self.new_subgoals)
        prob_feasible_dict = self.subgoal_property_net(
            datum=self.gcn_graph_input,
            subgoals=self.subgoals
        )
        if self.normalize:
            # normalize the probabilities
            total_prob = 0
            for subgoal in self.subgoals:
                total_prob += prob_feasible_dict[subgoal]
            total_prob = min(total_prob, 1)
            for subgoal in self.subgoals:
                prob_feasible_dict[subgoal] /= total_prob
        if self.assisstance:
            # Make all subgoals not in target container rooms infeasible
            # find the room of the target containers
            all_target = []
            for obj_idx in self.partial_map.obj_node_idx:
                if self.partial_map.org_node_names[obj_idx] == \
                   self.partial_map.org_node_names[self.partial_map.target_obj]:
                    all_target.append(obj_idx)
            all_target_containers = {self.partial_map.org_edge_index[0][self.partial_map.org_edge_index[1].index(obj_idx)]
                                     for obj_idx in all_target}
            target_container = [subgoal.value
                                for subgoal in self.subgoals
                                if subgoal in all_target_containers]
            target_rooms = set()
            for container in target_container:
                cnt_room = self.partial_map.org_edge_index[0][self.partial_map.org_edge_index[1].index(container)]
                target_rooms.add(cnt_room)
            for subgoal in self.subgoals:
                sub_room = self.partial_map.org_edge_index[0][self.partial_map.org_edge_index[1].index(subgoal.value)]
                if sub_room not in target_rooms:
                    prob_feasible_dict[subgoal] = 0

        for subgoal in self.subgoals:
            subgoal.set_props(
                prob_feasible=prob_feasible_dict[subgoal])
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:6.4f}|'
                    f'at= {self.partial_map.org_node_names[subgoal.value]}'
                )

        if self.verbose:
            print(" ")

    def compute_selected_subgoal(self):
        min_cost, frontier_ordering = (
            taskplan.core.get_best_expected_cost_and_frontier_list(
                self.subgoals,
                self.partial_map,
                self.robot_pose,
                self.destination,
                num_frontiers_max=NUM_MAX_FRONTIERS,
                alternate_sampling=True))
        return frontier_ordering[0]
