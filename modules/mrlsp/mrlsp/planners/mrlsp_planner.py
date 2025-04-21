import gridmap
from .planner import BaseMRLSPPlanner
import lsp
import torch
from mrlsp.utils.utility import (get_top_n_frontiers_multirobot,
                                 get_multirobot_distances,
                                 find_action_list_from_cost_matrix_using_lsa)
import mrlsp

NUM_MAX_FRONTIERS = 9


class MRKnownSubgoalPlanner(BaseMRLSPPlanner):
    '''Multi-robot Known Subgoal Planner'''
    def __init__(self, robots, goal, known_map, args):
        super(MRKnownSubgoalPlanner, self).__init__(robots, goal, args)
        self.known_map = known_map
        self.inflated_known_grid = gridmap.utils.inflate_grid(
            known_map, inflation_radius=self.inflation_radius)

    def _update_subgoal_properties(self, robot_pose, goal_pose):
        new_subgoals = [s for s in self.subgoals if not s.props_set]
        lsp.core.update_frontiers_properties_known(self.inflated_known_grid,
                                                   self.inflated_grid,
                                                   self.subgoals, new_subgoals,
                                                   robot_pose[0], goal_pose,
                                                   self.downsample_factor)

        for s in self.subgoals:
            if s.prob_feasible == 0.0:
                s.prob_feasible = 0.0001
            if s.prob_feasible == 1.0:
                s.prob_feasible = 0.9999

        print("All subgoals")
        for subgoal in self.subgoals:
            if not self.args.silence and subgoal.prob_feasible > 0.0:
                print(" " * 20 + "PLAN  (%.2f %.2f) | %.6f | %7.2f | %7.2f" %
                      (subgoal.get_centroid()[0], subgoal.get_centroid()[1],
                       subgoal.prob_feasible, subgoal.delta_success_cost,
                       subgoal.exploration_cost))

    def compute_selected_subgoal(self):
        # If goal in range, return None as action
        if lsp.core.goal_in_range(grid=self.observed_map,
                                  robot_pose=None, goal_pose=self.goal, frontiers=self.subgoals):
            joint_action = [None for i in range(len(self.robots))]
            return joint_action

        _, joint_action = mrlsp.core.get_best_expected_ordering_and_cost(self.inflated_grid,
                                                                            self.robots,
                                                                            self.goal,
                                                                            self.subgoals)
        return joint_action

    def _update_subgoal_inputs(self, images, robot_poses, goal_pose):
        pass


class MRLearnedSubgoalPlanner(BaseMRLSPPlanner):
    '''Multi-robot Learned Subgoal Planner'''

    def __init__(self, robots, goal, args, device=None, verbose=False):
        super(MRLearnedSubgoalPlanner, self).__init__(robots, goal, args)

        if device is not None:
            self.device = device
        else:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")

        self.subgoal_property_net = lsp.learning.models.VisLSPOriented.get_net_eval_fn(
            args.network_file, device=self.device)
        self.verbose = verbose

    def compute_selected_subgoal(self):
        # If goal in range, return None as action
        if lsp.core.goal_in_range(grid=self.observed_map,
                                  robot_pose=None, goal_pose=self.goal, frontiers=self.subgoals):
            joint_action = [None for i in range(len(self.robots))]
            return joint_action

        _, joint_action = mrlsp.core.get_best_expected_ordering_and_cost(self.inflated_grid,
                                                                         self.robots,
                                                                         self.goal,
                                                                         self.subgoals,
                                                                         num_frontiers_max=NUM_MAX_FRONTIERS,
                                                                         verbose=self.verbose)
        return joint_action

    def _update_subgoal_properties(self, robot_poses, goal_pose):

        for subgoal in self.subgoals:
            if subgoal.props_set:
                continue

            [prob_feasible, delta_success_cost, exploration_cost] = \
                self.subgoal_property_net(subgoal.nn_input_data)

            # sometimes the network returns negative values, which is not possible
            if delta_success_cost < 0:
                delta_success_cost = 0
            if exploration_cost < 0:
                exploration_cost = 0

            subgoal.set_props(prob_feasible=prob_feasible,
                              delta_success_cost=delta_success_cost,
                              exploration_cost=exploration_cost)

        # for subgoal in self.subgoals:
        #     if not self.args.silence and subgoal.prob_feasible > 0.0:
        #         print(" " * 20 + "PLAN  (%.2f %.2f) | %.6f | %7.2f | %7.2f" %
        #               (subgoal.get_centroid()[0], subgoal.get_centroid()[1],
        #                subgoal.prob_feasible, subgoal.delta_success_cost,
        #                subgoal.exploration_cost))
