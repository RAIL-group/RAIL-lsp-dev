import copy
import numpy as np
import itertools
import mrlsp_accel
from mr_exploration.utils.utility import (get_top_n_frontiers_multirobot,
                                 get_multirobot_distances)


class MRFState(object):
    def __init__(self, new_frontier=None, old_state=None, robots=None, distances=None, all_subgoals=None):
        if old_state is not None:
            self.prev_robot_locations = copy.copy(old_state.target_robot_locations)
            self.target_robot_locations = copy.copy(old_state.target_robot_locations)
            self.cost_to_target = copy.copy(old_state.cost_to_target)
            self.frontier_list = copy.copy(old_state.frontier_list)
            self.hash_to_subgoal = old_state.hash_to_subgoal
            self.distances = old_state.distances
            self.num_robots = old_state.num_robots

            done_robot_id = np.argmin(old_state.cost_to_target - old_state.progress)
            delta_t = (old_state.cost_to_target - old_state.progress)[done_robot_id]
            delta_fi = old_state.target_robot_locations[done_robot_id]

            ps = self.hash_to_subgoal[delta_fi].prob_feasible if delta_fi in self.hash_to_subgoal else 0
            Rs = self.distances['goal'][delta_fi] + \
                self.hash_to_subgoal[delta_fi].delta_success_cost if delta_fi in self.hash_to_subgoal else 0
            Re = self.hash_to_subgoal[delta_fi].exploration_cost if delta_fi in self.hash_to_subgoal else 0
            Q_success = Rs - min(Rs, Re)
            self.cost = old_state.cost + old_state.prob * (delta_t + ps * Q_success)
            self.prob = old_state.prob * (1 - ps)
            # store and update the unexplored subgoals
            self.unexplored = {u_poi for u_poi in old_state.unexplored
                               if not u_poi == old_state.target_robot_locations[done_robot_id]}

            self.progress = copy.copy(old_state.progress) + delta_t
            self.progress[done_robot_id] = 0


            self.prev_robot_locations[done_robot_id] = self.target_robot_locations[done_robot_id]
            if new_frontier is not None:
                # Don't assign when the frontier list is equal to the overall subgoals
                # Except when the number of subgoal is less than the number of robots assign the new frontier
                if (len(self.frontier_list) < len(self.hash_to_subgoal) or
                        len(self.frontier_list) < self.num_robots):
                    self.frontier_list +=  [hash(new_frontier)]
                self.target_robot_locations[done_robot_id] = hash(new_frontier)
                self.cost_to_target[done_robot_id] = self.get_cost_to_target(self.prev_robot_locations[done_robot_id],
                                                                            self.target_robot_locations[done_robot_id])

                # If any target_robot_locations is not in unexplored, then it is already explored
                # assign it to nearest unexplored location
                already_explored_frontiers = [(idx, f) for idx, f in enumerate(self.target_robot_locations)
                                            if f not in self.unexplored and f in self.hash_to_subgoal]
                for idx, exp_f in already_explored_frontiers:
                    new_target, distance = min([(f, self.get_cost_to_target(exp_f, f)) for f in self.unexplored],
                                                key=lambda x: x[1])
                    self.cost_to_target[idx] += distance
                    self.target_robot_locations[idx] = new_target
        else:
            self.frontier_list = []
            self.distances = distances
            self.num_robots = len(robots)
            self.progress = np.array([0.0] * self.num_robots)
            self.cost_to_target = np.array([0.0] * self.num_robots)
            self.prev_robot_locations = np.array(robots)
            self.target_robot_locations = np.array(robots)
            self.unexplored = {hash(s) for s in all_subgoals}
            self.unassigned_subgoals = [hash(s) for s in all_subgoals]
            self.hash_to_subgoal = {hash(s): s for s in all_subgoals}
            self.prob = 1
            self.cost = 0

    def __lt__(self, other):
        return self.cost < other.cost

    def get_cost_to_target(self, from_frontier, to_frontier):
        frontier_return_cost = (min(self.distances['goal'][from_frontier]
                                    + self.hash_to_subgoal[from_frontier].delta_success_cost,
                                    self.hash_to_subgoal[from_frontier].exploration_cost)
                                if from_frontier in self.hash_to_subgoal else 0)
        kc = self.distances['all'][frozenset([from_frontier, to_frontier])]
        Rs = self.distances['goal'][to_frontier] + \
            self.hash_to_subgoal[to_frontier].delta_success_cost if to_frontier in self.hash_to_subgoal else 0
        Re = self.hash_to_subgoal[to_frontier].exploration_cost if to_frontier in self.hash_to_subgoal else 0
        knowledge_time = min(Rs,Re)
        return frontier_return_cost + kc + knowledge_time

def mr_fstate_planner_compute_end_assignment(mr_fstate):
    remaining_robot_assignment = len([f for f in mr_fstate.target_robot_locations if f not in mr_fstate.hash_to_subgoal])
    num_steps = len(mr_fstate.unexplored) + remaining_robot_assignment
    for i in range(num_steps):
        delta_fi = mr_fstate.target_robot_locations[np.argmin(mr_fstate.cost_to_target - mr_fstate.progress)]
        remaining_subgoals = sorted([(f, mr_fstate.get_cost_to_target(delta_fi, f))
                                    for f in mr_fstate.unexplored if f != delta_fi], key=lambda x: x[1])
        new_target = mr_fstate.hash_to_subgoal[remaining_subgoals[0][0]] if len(remaining_subgoals) > 0 else None
        mr_fstate = MRFState(new_target, mr_fstate)
    return mr_fstate

def get_distances_for_mrfstate(subgoals, robots_hash_dict, distances):
    distances_py = {}
    distances_py['goal'] = {hash(s): distances['goal'][s] for s in subgoals}
    distances_py['all'] = {
        frozenset([hash(sp[0]), hash(sp[1])]): distances['frontier'][frozenset([sp[0], sp[1]])]
        for sp in itertools.permutations(subgoals, 2) if sp[0] != sp[1]
    }
    distances_py['all'].update({
        frozenset([robots_hash_dict[r], hash(s)]): distances['robot'][i][s]
        for i, r in enumerate(robots_hash_dict) for s in subgoals
    })
    return distances_py

def get_mr_ordering_cost_py(robots_hash, subgoals, distances):
    mr_fstate = MRFState(robots=robots_hash, distances=distances, all_subgoals=subgoals)
    for i, s in enumerate(subgoals):
        mr_fstate = MRFState(s, mr_fstate)
    mr_fstate = mr_fstate_planner_compute_end_assignment(mr_fstate)
    return mr_fstate.cost


def get_mr_ordering_cost_cpp(robots_hash, subgoals, distances):
    subgoals_cpp = [
        mrlsp_accel.SubgoalData(s.prob_feasible,
                                          s.delta_success_cost,
                                          s.exploration_cost,
                                          hash(s)) for s in subgoals
    ]
    gd_cpp = {hash(s): distances['goal'][hash(s)] for s in subgoals}
    inter_distances_cpp = {
        (hash(sp[0]), hash(sp[1])): distances['all'][frozenset([hash(sp[0]), hash(sp[1])])]
        for sp in itertools.permutations(robots_hash + subgoals, 2)
    }
    return mrlsp_accel.get_mr_ordering_cost(robots_hash, subgoals_cpp, gd_cpp, inter_distances_cpp)


def get_mr_lowest_cost_ordering_py(robots_hash, frontiers, distances):
    '''Recursively compute the lowest cost ordering of provided frontiers'''
    def get_ordering_sub(frontiers, state):
        if len(frontiers) == 1:
            s = MRFState(frontiers[0], state)
            s = mr_fstate_planner_compute_end_assignment(s)
            get_ordering_sub.bound = min(get_ordering_sub.bound, s.cost)
            return s

        if state.cost > get_ordering_sub.bound:
            return state

        return min([
            get_ordering_sub([fn for fn in frontiers if fn != f],
                            MRFState(f, state))
            for f in frontiers
        ])
    get_ordering_sub.bound = 1e10
    state = MRFState(robots=robots_hash, distances=distances, all_subgoals=frontiers)
    state = get_ordering_sub(frontiers, state)
    return state.cost, state.frontier_list


def get_mr_lowest_cost_ordering_cpp(robots_hash, subgoals, distances):
    subgoals_cpp = [
        mrlsp_accel.SubgoalData(s.prob_feasible,
                                s.delta_success_cost,
                                s.exploration_cost,
                                hash(s)) for s in subgoals
    ]
    gd_cpp = {hash(s): distances['goal'][hash(s)] for s in subgoals}
    inter_distances_cpp = {
        (hash(sp[0]), hash(sp[1])): distances['all'][frozenset([hash(sp[0]), hash(sp[1])])]
        for sp in itertools.permutations(subgoals, 2) if sp[0] != sp[1]
    }
    inter_distances_cpp.update({
        (r, hash(s)): distances['all'][frozenset([r, hash(s)])]
        for r in robots_hash for s in subgoals
    })
    return mrlsp_accel.get_mr_lowest_cost_ordering(robots_hash, subgoals_cpp, gd_cpp, inter_distances_cpp)


def get_best_expected_ordering_and_cost(inflated_grid,
                                        robots,
                                        goal,
                                        subgoals,
                                        num_frontiers_max,
                                        use_py=False,
                                        verbose=True):
    # for cpp
    num_robots = len(robots)
    distances_mr = get_multirobot_distances(
         inflated_grid, robots, [goal], subgoals)
    subgoals = get_top_n_frontiers_multirobot(num_robots,
                                              subgoals,
                                              distances_mr,
                                              n=num_frontiers_max)
    unexplored_frontiers = list(subgoals)

    if verbose:
        print("Top n subgoals")
        for subgoal in unexplored_frontiers:
            if subgoal.prob_feasible > 0.0:
                print(" " * 20 + "PLAN  (%.2f %.2f) | %.6f | %7.2f | %7.2f" %
                      (subgoal.get_centroid()[0], subgoal.get_centroid()[1],
                       subgoal.prob_feasible, subgoal.delta_success_cost,
                       subgoal.exploration_cost))

    s_dict = {hash(s): s for s in unexplored_frontiers}
    robots_hash_dict = {r: hash(r) for r in robots}
    robots_hash = [hash(r) for r in robots]
    distances_py = get_distances_for_mrfstate(
            unexplored_frontiers, robots_hash_dict, distances_mr)
    if not use_py:
        cost, ordering = get_mr_lowest_cost_ordering_cpp(robots_hash,
                                                         unexplored_frontiers,
                                                         distances_py)
    else:
        cost, ordering = get_mr_lowest_cost_ordering_py(robots_hash,
                                                        unexplored_frontiers,
                                                        distances_py)

    ordering = [s_dict[s] for s in ordering]
    joint_action = ordering[:num_robots]
    return cost, joint_action
