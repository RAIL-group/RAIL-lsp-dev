import copy
# import numpy as np
import logging
import time
# import torch

import lsp
import taskplan
from procint.learning.models.fcnn import IntvFcnn
from taskplan.planners.planner import NUM_MAX_FRONTIERS
from taskplan.core import get_subgoal_distances, get_robot_distances


class IntvPlanner(taskplan.planners.planner.LearnedPlanner):
    def __init__(self, args, partial_map, device=None, verbose=True):
        super(IntvPlanner, self).__init__(
            args, partial_map, device, verbose)

        self.subgoal_property_net, self.model = IntvFcnn.get_net_eval_fn(
            args.network_file, device=self.device, do_return_model=True)

    def compute_selected_subgoal(self, return_cost=False):
        min_cost, frontier_ordering = (
            taskplan.core.get_best_expected_cost_and_frontier_list(
                self.subgoals,
                self.partial_map,
                self.robot_pose, None,
                num_frontiers_max=NUM_MAX_FRONTIERS))
        if return_cost:
            return min_cost, frontier_ordering[0]
        return frontier_ordering[0]

    # def compute_backup_subgoal(self, selected_subgoal):
    #     subgoals, distances = self.get_subgoals_and_distances()
    #     return lsp.core.get_lowest_cost_ordering_not_beginning_with(
    #         selected_subgoal, subgoals, distances)[1][0]

    def compute_subgoal_data(self,
                             chosen_subgoal,
                             num_frontiers_max=NUM_MAX_FRONTIERS,
                             do_return_ind_dict=False):
        # Compute chosen frontier
        logger = logging.getLogger("IntvPlanner")
        stime = time.time()
        policy_data, subgoal_ind_dict = get_policy_data_for_frontiers(
            self.partial_map,
            self.robot_pose,
            chosen_subgoal,
            self.subgoals,
            num_frontiers_max=num_frontiers_max)
        logger.debug(f"time to get policy data: {time.time() - stime}")

        if do_return_ind_dict:
            return policy_data, subgoal_ind_dict
        else:
            return policy_data

    # def get_subgoals_and_distances(self, subgoals_of_interest=[]):
    #     """Helper function for getting data."""
    #     # Remove frontiers that are infeasible
    #     subgoals = [s for s in self.subgoals if s.prob_feasible > 0]
    #     subgoals = list(set(subgoals) | set(subgoals_of_interest))

    #     # Calculate the distance to the goal and to the robot.
    #     goal_distances = get_goal_distances(
    #         self.inflated_grid,
    #         self.goal,
    #         frontiers=subgoals,
    #         downsample_factor=self.downsample_factor)

    #     robot_distances = get_robot_distances(
    #         self.inflated_grid,
    #         self.robot_pose,
    #         frontiers=subgoals,
    #         downsample_factor=self.downsample_factor)

    #     # Get the most n probable frontiers to limit computational load
    #     if NUM_MAX_FRONTIERS > 0 and NUM_MAX_FRONTIERS < len(subgoals):
    #         subgoals = get_top_n_frontiers(subgoals, goal_distances,
    #                                        robot_distances, NUM_MAX_FRONTIERS)
    #         subgoals = list(set(subgoals) | set(subgoals_of_interest))

    #     # Calculate robot and frontier distances
    #     frontier_distances = get_frontier_distances(
    #         self.inflated_grid,
    #         frontiers=subgoals,
    #         downsample_factor=self.downsample_factor)

    #     distances = {
    #         'frontier': frontier_distances,
    #         'robot': robot_distances,
    #         'goal': goal_distances,
    #     }

    #     return subgoals, distances

    def generate_counterfactual_explanation(self,
                                            query_subgoal,
                                            limit_num=-1,
                                            do_freeze_selected=True,
                                            keep_changes=False,
                                            margin=0,
                                            learning_rate=1.0e-4):
        # Initialize the datum
        device = self.device
        chosen_subgoal = self.compute_selected_subgoal()
        datum, subgoal_ind_dict = self.compute_subgoal_data(
            chosen_subgoal, NUM_MAX_FRONTIERS, do_return_ind_dict=True)
        datum['subgoal_data'] = taskplan.utilities.utils.preprocess_fcnn_data(
            datum['subgoal_data'])
        datum = self.model.update_datum(datum, device)

        # Now we want to rearrange things a bit: the new 'target' subgoal we set to
        # our query_subgoal and we populate the 'backup'
        # subgoal with the 'chosen' subgoal (the subgoal the agent actually chose).
        datum['target_subgoal_ind'] = subgoal_ind_dict[query_subgoal]
        if do_freeze_selected:
            datum['backup_subgoal_ind'] = subgoal_ind_dict[chosen_subgoal]

        # We update the datum to reflect this change (and confirm it worked).
        datum = self.model.update_datum(datum, device)

        # Compute the 'delta subgoal data'. This is how we determine the
        # 'importance' of each of the subgoal properties. In practice, we will sever
        # the gradient for all but a handful of these with an optional parameter
        # (not set here).
        base_model_state = self.model.state_dict(keep_vars=False)
        base_model_state = copy.deepcopy(base_model_state)
        base_model_state = {k: v.cpu() for k, v in base_model_state.items()}

        updated_datum, base_subgoal_props, updated_subgoal_props = (
            self.model.update_model_counterfactual(datum, limit_num,
                                                   margin, learning_rate,
                                                   self.device))

        # Restore the model to its previous value
        if not keep_changes:
            print("Restoring Model")
            self.model.load_state_dict(base_model_state)
            self.model.eval()
            self.model = self.model.to(device)
        else:
            print("Keeping model")
            # self._recompute_all_subgoal_properties()
            self._update_subgoal_properties()

    # def plot_map_with_plan(self, ax=None, robot_poses=None, image=None,
    #                        query_subgoal=None, datum=None, subgoal_props=None, subgoal_ind_dict=None):
    #     import matplotlib.pyplot as plt
    #     ax_img = plt.subplot(121)
    #     ax_img.axes.xaxis.set_visible(False)
    #     ax_img.axes.yaxis.set_visible(False)

    #     # Initialize the datum
    #     device = self.device
    #     chosen_subgoal = self.compute_selected_subgoal()

    #     if chosen_subgoal is None:
    #         lsp_xai.utils.plotting.plot_map(
    #             ax_img, self, robot_poses=robot_poses)
    #         return

    #     if datum is None or subgoal_props is None:
    #         datum, subgoal_ind_dict = self.compute_subgoal_data(
    #             chosen_subgoal, 24, do_return_ind_dict=True)
    #         datum = self.model.update_datum(datum, device)
    #         delta_subgoal_data = self.model.get_subgoal_prop_impact(
    #             datum, device, delta_cost_limit=-1e10)

    #         # Compute the subgoal props
    #         nn_out, ind_mapping = self.model(datum, device)
    #         is_feasibles = torch.nn.Sigmoid()(nn_out[:, 0])
    #         delta_success_costs = nn_out[:, 1]
    #         exploration_costs = nn_out[:, 2]
    #         subgoal_props, _, _ = self.model.compute_subgoal_props(
    #             is_feasibles,
    #             delta_success_costs,
    #             exploration_costs,
    #             datum['subgoal_data'],
    #             ind_mapping,
    #             device,
    #             limit_subgoals_num=0,
    #             delta_subgoal_data=delta_subgoal_data)

    #     policy = datum['target_subgoal_policy']['policy']

    #     lsp_xai.utils.plotting.plot_map_with_plan(
    #         ax_img, self, subgoal_ind_dict, policy, subgoal_props,
    #         robot_poses=robot_poses)

    #     # Plot the onboard image
    #     if image is not None:
    #         ax = plt.subplot(3, 2, 2)
    #         ax.imshow(image)
    #         ax.set_title('Onboard Image')
    #         ax.axes.xaxis.set_visible(False)
    #         ax.axes.yaxis.set_visible(False)

    #     # Plt the chosen subgoal
    #     ax = plt.subplot(3, 2, 4)
    #     chosen_subgoal_ind = subgoal_ind_dict[chosen_subgoal]
    #     ax.imshow(datum['subgoal_data'][chosen_subgoal_ind]['image'])
    #     pf_chosen = subgoal_props[chosen_subgoal_ind].prob_feasible
    #     ax.set_title(f'Subgoal 0: $P_S$ = {pf_chosen*100:.1f}\\%')
    #     ax.axes.xaxis.set_visible(False)
    #     ax.axes.yaxis.set_visible(False)
    #     ax.plot([0.5, 0.5], [1.0, 0.0],
    #             transform=ax.transAxes,
    #             color=[0, 0, 1],
    #             alpha=0.3)

    #     # Plot the query/backup subgoal
    #     ax = plt.subplot(3, 2, 6)
    #     if query_subgoal is None:
    #         query_subgoal = self.compute_backup_subgoal(chosen_subgoal)
    #     query_subgoal_ind = subgoal_ind_dict[query_subgoal]
    #     ax.imshow(datum['subgoal_data'][query_subgoal_ind]['image'])
    #     pf_query = subgoal_props[query_subgoal_ind].prob_feasible
    #     ax.set_title(f'Subgoal 1: $P_S$ = {pf_query*100:.1f}\\%')
    #     ax.axes.xaxis.set_visible(False)
    #     ax.axes.yaxis.set_visible(False)
    #     ax.plot([0.5, 0.5], [1.0, 0.0],
    #             transform=ax.transAxes,
    #             color=[0, 0, 1],
    #             alpha=0.3)

    # @classmethod
    # def create_with_state(cls, planner_state_datum, network_file):
    #     # Initialize the planner
    #     args = planner_state_datum['args']
    #     goal = planner_state_datum['goal']
    #     args.network_file = network_file
    #     planner = cls(goal, args)

    #     planner.subgoals = planner_state_datum['subgoals']
    #     planner.observed_map = planner_state_datum['observed_map']
    #     planner.inflated_grid = planner_state_datum['inflated_grid']
    #     planner._recompute_all_subgoal_properties()

    #     return planner


# Alt versions of functions
def get_policy_data_for_frontiers(partial_map,
                                  robot_pose,
                                  chosen_frontier,
                                  all_frontiers,
                                  num_frontiers_max=0):
    """Compute the optimal orderings for each frontier of interest and return a data
    structure containing all the information that would be necessary to compute the
    expected cost for each. Also returns the mapping from 'frontiers' to 'inds'."""

    # Remove frontiers that are infeasible
    frontiers = [f for f in all_frontiers if f.prob_feasible != 0]
    subgoals = list(set(frontiers) | set([chosen_frontier]))

    # Calculate the distance to the goal, if infeasible, remove frontier
    goal_distances = {subgoal: 0 for subgoal in subgoals}

    robot_distances = get_robot_distances(
        partial_map.grid, robot_pose, subgoals)

    # Get the most n probable frontiers to limit computational load
    if num_frontiers_max > 0 and num_frontiers_max < len(subgoals):
        frontiers = lsp.core.get_top_n_frontiers_distance(
            subgoals, goal_distances, robot_distances, num_frontiers_max)
        subgoals = list(set(frontiers) | set([chosen_frontier]))

    # Calculate robot and frontier distances
    subgoal_distances = get_subgoal_distances(partial_map.grid, subgoals)

    subgoal_ind_dict = {f: ind for ind, f in enumerate(subgoals)}
    robot_distances_ind = {
        subgoal_ind_dict[f]: robot_distances[f]
        for f in subgoals
    }
    goal_distances_ind = {
        subgoal_ind_dict[f]: goal_distances[f]
        for f in subgoals
    }
    subgoal_distances_ind = {}
    for ind, f1 in enumerate(subgoals[:-1]):
        f1_ind = subgoal_ind_dict[f1]
        for f2 in subgoals[ind + 1:]:
            f2_ind = subgoal_ind_dict[f2]
            subgoal_distances_ind[frozenset(
                [f1_ind, f2_ind])] = (subgoal_distances[frozenset([f1, f2])])

    if subgoal_distances is not None:
        assert len(subgoal_distances.keys()) == len(
            subgoal_distances_ind.keys())

    # Finally, store the data relevant for
    # estimating the frontier properties
    # for gnn we take the subgoal node positions on the partial map
    subgoals_idx = [s.value for s in subgoals]
    graph, _ = partial_map.update_graph_and_subgoals(subgoals_idx)
    # gnn_input_graph = partial_map.prepare_gcn_input(graph, subgoals_idx)
    gnn_input_graph = partial_map.prepare_fcnn_input(subgoals_idx)

    # subgoal_data = {
    #     ind: f.nn_input_data
    #     for f, ind in subgoal_distances_ind.items()
    # }

    return {
        'subgoal_data': gnn_input_graph,
        'distances': {
            'frontier': subgoal_distances_ind,
            'robot': robot_distances_ind,
            'goal': goal_distances_ind,
        },
        'target_subgoal_ind': subgoal_ind_dict[chosen_frontier]
    }, subgoal_ind_dict
