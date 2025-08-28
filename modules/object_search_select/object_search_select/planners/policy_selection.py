from object_search.planners.planner import Planner
from object_search_select import offline_replay
import numpy as np
import object_search


class PolicySelectionPlanner(Planner):
    """Meta-planner class that handles selection among multiple planners/policies."""
    def __init__(self, target_obj_info, planners, chosen_planner_idx, args, verbose=True):
        super(PolicySelectionPlanner, self).__init__(target_obj_info, args, verbose)
        self.planners = planners
        self.chosen_planner_idx = chosen_planner_idx
        self.navigation_data = {
            'target_obj_info': target_obj_info,
            'graph': [],
            'subgoals': [],
        }

    def update(self, graph, grid, subgoals, robot_pose):
        """Updates the information in currently chosen planner"""
        self.graph = graph
        self.grid = grid
        self.robot_pose = robot_pose
        self.planners[self.chosen_planner_idx].update(graph, grid, subgoals, robot_pose)
        self.subgoals = self.planners[self.chosen_planner_idx].subgoals

        self.navigation_data['graph'].append(graph)
        self.navigation_data['subgoals'].append(subgoals)

    def compute_selected_subgoal(self):
        """Compute selected subgoal from the chosen planner."""
        return self.planners[self.chosen_planner_idx].compute_selected_subgoal()

    def get_costs(self, robot):
        """ After navigation is complete, get replayed costs for all other planners."""
        self.navigation_data['final_partial_grid'] = self.grid
        self.navigation_data['final_subgoals'] = self.subgoals
        self.navigation_data['robot_path'] = robot.all_poses
        self.navigation_data['robot_pose'] = robot.all_poses[0]
        net_motion, _ = object_search.utils.compute_cost_and_trajectory(self.grid, robot.all_poses,
                                                                        self.args.resolution)
        self.navigation_data['net_motion'] = net_motion

        lb_costs = np.full((len(self.planners), 2), np.nan)
        planner_costs = np.full(len(self.planners), np.nan)
        self.args.chosen_planner_idx = self.chosen_planner_idx
        for i, planner in enumerate(self.planners):
            self.args.replayed_planner_idx = i
            if i == self.chosen_planner_idx:
                # The cost of the chosen planner is the net distance traveled
                planner_costs[i] = self.navigation_data['net_motion']
            else:
                # For other planners, get lower bound costs via offline replay
                optimistic_lb, simply_connected_lb = offline_replay.get_lowerbound_planner_costs(self.navigation_data,
                                                                                                 planner,
                                                                                                 self.args)
                lb_costs[i] = [optimistic_lb, simply_connected_lb]

        return planner_costs, lb_costs
