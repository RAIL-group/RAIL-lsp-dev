import torch
from .planner import Planner
from object_search.learning import utils
from object_search.learning.models.fcnn import FCNN
import lsp

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
        subgoals = [s for s in self.subgoals if s.prob_feasible != 0]

        # Get robot distances
        robot_distances = self.get_robot_distances(
            self.grid, self.robot_pose, subgoals)

        # Get goal distances
        if self.destination is None:
            goal_distances = {subgoal: robot_distances[subgoal]
                              for subgoal in subgoals}
        else:
            goal_distances = self.get_robot_distances(
                self.grid, self.destination, subgoals)

        # Get most probable n subgoals to limit computational load
        if NUM_MAX_FRONTIERS > 0 and NUM_MAX_FRONTIERS < len(subgoals):
            subgoals = lsp.core.get_top_n_frontiers(subgoals, goal_distances,
                                                    robot_distances, NUM_MAX_FRONTIERS)

        # Calculate robot and subgoal distances
        frontier_distances = self.get_subgoal_distances(self.grid, subgoals)

        distances = {
            'frontier': frontier_distances,
            'robot': robot_distances,
            'goal': goal_distances,
        }

        min_cost, frontier_ordering = lsp.core.get_lowest_cost_ordering(subgoals, distances)
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
