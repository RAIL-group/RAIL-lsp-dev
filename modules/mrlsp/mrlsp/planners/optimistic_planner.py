import numpy as np
from .planner import MRPlanner
from mrlsp.utils.utility import get_multirobot_distances, find_action_list_from_cost_matrix_using_lsa


class MROptimisticPlanner(MRPlanner):
    def __init__(self, robots, goal_poses, args):
        super(MROptimisticPlanner, self).__init__(robots, goal_poses, args)
        self.inflation_radius = args.inflation_radius_m / args.base_resolution
        self.joint_action = [None for i in range(self.num_robots)]

    def compute_selected_subgoal(self):
        '''TODO: Right now, goal pose is sent as list: Just because every other function use list of goalsl,
        The functionality that is thought to be extended, where goal is multiple'''
        distances_mr = get_multirobot_distances(self.inflated_grid, self.robots, [self.goal], self.subgoals)
        # find cost for robot to reach goal from all the subgoal, and create a cost matrix.
        cost_dictionary = [None for _ in range(len(self.robots))]
        for i in range(len(self.robots)):
            cost_dictionary[i] = {
                subgoal: distances_mr['robot'][i][subgoal] + distances_mr['goal'][subgoal]
                for subgoal in self.subgoals
            }
        subgoal_matrix = np.array([list(cd.keys()) for cd in cost_dictionary])
        cost_matrix = np.array([list(cd.values()) for cd in cost_dictionary])
        # from the cost matrix, return the list of subgoals that has the least cost.
        joint_action = find_action_list_from_cost_matrix_using_lsa(cost_matrix, subgoal_matrix)

        return joint_action
