import lsp
# from mrlsp.utils.utility import (get_top_n_frontiers_multirobot,
#                                  get_multirobot_distances,
#                                  find_action_list_from_cost_matrix_using_lsa)
from mrlsp.planners import MRLearnedSubgoalPlanner
import mrlsp

NUM_MAX_FRONTIERS = 9


class MRLSPInfoGatherPlanner(MRLearnedSubgoalPlanner):

    def compute_selected_subgoal(self):
        # If goal in range, return None as action
        if lsp.core.goal_in_range(grid=self.observed_map,
                                  robot_pose=None, goal_pose=self.goal, frontiers=self.subgoals):
            joint_action = [None for i in range(len(self.robots))]
            return joint_action

        cost, joint_action = mrlsp.core.get_best_expected_ordering_and_cost(
            self.inflated_grid,
            self.robots,
            self.goal,
            self.subgoals,
            num_frontiers_max=NUM_MAX_FRONTIERS
        )
        print(f"Cost: {cost}")
        return joint_action
