import numpy as np
import lsp


class Subgoal:
    def __init__(self, id, pose=None):
        self.id = id
        self.pose = pose
        self.points = np.array([[int(pose[0])],
                                [int(pose[1])]])
        self.props_set = False
        self.is_from_last_chosen = False
        self.is_obstructed = False
        self.prob_feasible = 1.0
        self.delta_success_cost = 0.0
        self.exploration_cost = 0.0
        self.negative_weighting = 0.0
        self.positive_weighting = 0.0

        self.counter = 0
        self.hash = hash(self.id)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return f"<Subgoal {self.id} at {self.pose}>"

    def get_frontier_point(self):
        return np.array([self.pose[0], self.pose[1]], dtype=int)

    def set_props(self,
                  prob_feasible,
                  is_obstructed=False,
                  delta_success_cost=0,
                  exploration_cost=0,
                  positive_weighting=0,
                  negative_weighting=0,
                  counter=0,
                  last_observed_pose=None,
                  did_set=True):
        self.props_set = did_set
        self.just_set = did_set
        self.prob_feasible = prob_feasible
        self.is_obstructed = is_obstructed
        self.delta_success_cost = delta_success_cost
        self.exploration_cost = exploration_cost
        self.positive_weighting = positive_weighting
        self.negative_weighting = negative_weighting
        self.counter = counter
        self.last_observed_pose = last_observed_pose


def get_best_expected_cost_and_frontier_list(grid,
                                             robot_pose,
                                             destination,
                                             frontiers,
                                             num_frontiers_max=0):
    # Remove frontiers that are infeasible
    frontiers = [f for f in frontiers if f.prob_feasible != 0]

    # Get robot distances
    robot_distances = lsp.core.get_robot_distances(
        grid, robot_pose, frontiers)

    # Get goal distances
    if destination is None:
        goal_distances = {frontier: robot_distances[frontier]
                          for frontier in frontiers}
    else:
        goal_distances = lsp.core.get_robot_distances(
            grid, destination, frontiers)

    # Get most probable n frontiers to limit computational load
    if num_frontiers_max > 0 and num_frontiers_max < len(frontiers):
        frontiers = lsp.core.get_top_n_frontiers(frontiers, goal_distances,
                                                 robot_distances, num_frontiers_max)

    # Calculate robot and frontier distances
    frontier_distances = lsp.core.get_frontier_distances(grid, frontiers)

    distances = {
        'frontier': frontier_distances,
        'robot': robot_distances,
        'goal': goal_distances,
    }

    out = lsp.core.get_lowest_cost_ordering(frontiers, distances)
    return out
