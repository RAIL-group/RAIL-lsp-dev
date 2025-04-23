# Maggie Coombs, 02/24/2025, the goal of this planner is to explore unseen space with an optimistic planner
# that is a planner that reduces the cost of the overall moves of the robots by essentially choosing the closest
# frontier to explore.  

import numpy as np
from .planner import MRPlanner
from mr_exploration.utils.utility import get_multirobot_distances, find_action_list_from_cost_matrix_using_lsa, get_subgoal_areas


class MRAreaExplorePlanner(MRPlanner):
    def __init__(self, robots, args):
        super(MRAreaExplorePlanner, self).__init__(robots, args)
        self.inflation_radius = args.inflation_radius_m / args.base_resolution
        self.joint_action = [None for i in range(self.num_robots)]


    def compute_selected_subgoal(self, known_map):
        '''TODO: Right now, goal pose is sent as list: just because every other function uses list of goals.
        The functionality is expected to be extended to support multiple goals.'''
        
        distances_mr = get_multirobot_distances(self.inflated_grid, self.robots, self.subgoals)
        # Find cost for robot to reach each subgoal, and create a cost matrix.

        # Find the area behind each subgoal from utility, and create a dictionary
        explorable_regions, labels, subgoal_to_label = get_subgoal_areas(
            self.subgoals, self.inflated_grid, known_map, self.inflation_radius
        )

        subgoal_mask = np.zeros_like(self.inflated_grid, dtype=np.uint8)
        for subgoal in self.subgoals:
            x, y = map(int, subgoal.centroid[:2])
            subgoal_mask[x, y] = 1

        dist_weight = 1.0
        area_weight = -5.0

        all_subgoals = list(self.subgoals)
        num_subgoals = len(all_subgoals)
        num_robots = len(self.robots)
        subgoals_idx = {sg: idx for idx, sg in enumerate(all_subgoals)}

        cost_matrix = np.full((num_robots, num_subgoals), fill_value=1e9)
        subgoal_matrix = np.empty((num_robots, num_subgoals), dtype=object)

        for i in range(num_robots):
            for subgoal in all_subgoals:
                distance = distances_mr['robot'][i][subgoal]
                x, y = map(int, subgoal.centroid[:2])
                label_id = subgoal_to_label.get(subgoal, None)
                area = explorable_regions.get(label_id, 0)
                cost = dist_weight * distance + area_weight * area

                j = subgoals_idx[subgoal]
                cost_matrix[i, j] = cost
                subgoal_matrix[i, j] = subgoal

                # Debugging print
                print(f"Robot {i} | Subgoal {x, y} | Dist: {distance:.2f} | Area: {area} | Cost: {cost:.2f}")

        # From the cost matrix, return the list of subgoals with the least cost
        joint_action = find_action_list_from_cost_matrix_using_lsa(cost_matrix, subgoal_matrix)

        return joint_action

# I modified the optimistic_planner.py file by removing the goal_poses part as well as the calculation involving the distance from the "goal pose"
#here, for the area based planner I'll need to modify the planner to select the frontier or subgoal that has the maximum area connected to it,
# should I be worried about the cost of it, or is just going for the max good enough for now?