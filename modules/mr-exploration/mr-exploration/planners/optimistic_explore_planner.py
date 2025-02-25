# Maggie Coombs, 02/24/2025, the goal of this planner is to explore unseen space with an optimistic planner
# that is a planner that reduces the cost of the overall moves of the robots by essentially choosing the closest
# frontier to explore.  

import numpy as np
from .planner import MRPlanner
from mrlsp.utils.utility import get_multirobot_distances, find_action_list_from_cost_matrix_using_lsa


class MROptimisticExplorePlanner(MRPlanner):
    def __init__(self, robots, args):
        super(MROptimisticExplorePlanner, self).__init__(robots, args)
        self.inflation_radius = args.inflation_radius_m / args.base_resolution
        self.joint_action = [None for i in range(self.num_robots)]

    def compute_selected_subgoal(self):
        '''TODO: Right now, goal pose is sent as list: Just because every other function use list of goalsl,
        The functionality that is thought to be extended, where goal is multiple'''
        distances_mr = get_multirobot_distances(self.inflated_grid, self.robots, self.subgoals)
        # find cost for robot to reach goal from all the subgoal, and create a cost matrix.
        cost_dictionary = [None for _ in range(len(self.robots))]
        for i in range(len(self.robots)):
            cost_dictionary[i] = {
                subgoal: distances_mr['robot'][i][subgoal]
                for subgoal in self.subgoals
            }
        subgoal_matrix = np.array([list(cd.keys()) for cd in cost_dictionary])
        cost_matrix = np.array([list(cd.values()) for cd in cost_dictionary])
        # from the cost matrix, return the list of subgoals that has the least cost.
        joint_action = find_action_list_from_cost_matrix_using_lsa(cost_matrix, subgoal_matrix)

        return joint_action

# I modified the optimistic_planner.py file by removing the goal_poses part as well as the calculation involving the distance from the "goal pose"