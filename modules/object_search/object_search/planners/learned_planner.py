import torch
from .planner import Planner, BaseFrontierPlanner
from object_search.learning import utils
from object_search.learning.models.fcnn import FCNN
from object_search import core

NUM_MAX_FRONTIERS = 8


class LearnedPlanner(Planner):
    '''This planner calculates subgoal properties using the learned network
    and then uses LSP approach to pick the best available action (subgoal).
    '''
    def __init__(self, target_obj_info, args, subgoal_property_net,
                 preprocess_input_fn, destination=None, verbose=True):
        super(LearnedPlanner, self).__init__(target_obj_info, args, verbose)
        self.destination = destination
        self.subgoal_property_net = subgoal_property_net
        self.preprocess_input_fn = preprocess_input_fn

    def _update_subgoal_properties(self):
        nn_input_data = self.preprocess_input_fn(
            graph=self.graph,
            subgoals=self.subgoals,
            target_obj_info=self.target_obj_info,
        )
        prob_feasible_dict = self.subgoal_property_net(
            datum=nn_input_data,
            subgoals=self.subgoals
        )
        for subgoal in self.subgoals:
            subgoal.set_props(
                prob_feasible=prob_feasible_dict[subgoal])
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:.3f} | '
                    f'at {self.graph.get_node_name_by_idx(subgoal.id)}'
                )

    def compute_selected_subgoal(self):
        min_cost, frontier_ordering = core.get_best_expected_cost_and_frontier_list(
            self.grid,
            self.robot_pose,
            self.destination,
            self.subgoals,
            num_frontiers_max=NUM_MAX_FRONTIERS)

        return frontier_ordering[0]


class LearnedPlannerFCNN(LearnedPlanner):
    def __init__(self, target_obj_info, args, destination=None, device=None, verbose=True):
        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

        subgoal_property_net = FCNN.get_net_eval_fn(args.network_file, device)
        preprocess_input_fn = utils.prepare_fcnn_input
        super(LearnedPlannerFCNN, self).__init__(target_obj_info,
                                                 args,
                                                 subgoal_property_net,
                                                 preprocess_input_fn,
                                                 destination,
                                                 verbose)


class LearnedFrontierPlanner(BaseFrontierPlanner):
    '''This planner calculates subgoal properties using the learned network
    and then uses LSP approach to pick the best available frontier or container.
    '''
    def __init__(self, target_obj_info, args, subgoal_property_net,
                 preprocess_input_fn, verbose=True):
        super(LearnedFrontierPlanner, self).__init__(target_obj_info, args, verbose)
        self.subgoal_property_net = subgoal_property_net
        self.preprocess_input_fn = preprocess_input_fn

    def _update_subgoal_properties(self):
        nn_input_data = self.preprocess_input_fn(
            graph=self.graph,
            grid=self.grid,
            subgoals=self.subgoals,
            target_obj_info=self.target_obj_info,
        )
        prob_feasible_dict = self.subgoal_property_net(
            datum=nn_input_data,
            subgoals=self.subgoals
        )
        for subgoal in self.subgoals:
            subgoal.set_props(
                prob_feasible=prob_feasible_dict[subgoal])
            if self.verbose:
                if isinstance(subgoal, core.Subgoal):
                    print(
                        f'Ps={subgoal.prob_feasible:.3f} | '
                        f'at container {self.graph.get_node_name_by_idx(subgoal.id)}'
                    )
                else:
                    print(
                        f'Ps={subgoal.prob_feasible:.3f} | '
                        f'at frontier {subgoal.get_frontier_point()} (room: {subgoal.room_name})'
                    )

    def compute_selected_subgoal(self):
        min_cost, frontier_ordering = core.get_best_expected_cost_and_frontier_list(
            self.grid,
            self.robot_pose,
            None,
            self.subgoals,
            num_frontiers_max=NUM_MAX_FRONTIERS)

        return frontier_ordering[0]


class LearnedFrontierPlannerFCNN(LearnedFrontierPlanner):
    def __init__(self, target_obj_info, args, device=None, verbose=True):
        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

        subgoal_property_net = FCNN.get_net_eval_fn(args.network_file, device)
        preprocess_input_fn = utils.prepare_fcnn_input_frontiers
        super(LearnedFrontierPlannerFCNN, self).__init__(target_obj_info,
                                                         args,
                                                         subgoal_property_net,
                                                         preprocess_input_fn,
                                                         verbose)
