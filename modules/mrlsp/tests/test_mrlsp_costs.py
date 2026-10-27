import numpy as np
import itertools
import pytest
import lsp
import mrlsp_accel
from mrlsp.core import (get_mr_ordering_cost_py,
                        get_mr_ordering_cost_cpp,
                        get_mr_lowest_cost_ordering_py,
                        get_mr_lowest_cost_ordering_cpp,
                        get_mr_lowest_cost_state_py)

class Robot():
    def __init__(self, id):
        self.id = id

    def __hash__(self):
        return self.id

def get_inter_distances_cpp_for_test(subgoals, robots, distances):
    inter_distances_cpp = {
        (hash(sp[0]), hash(sp[1])): distances['all'][frozenset([hash(sp[0]), hash(sp[1])])]
        for sp in itertools.permutations(subgoals, 2) if sp[0] != sp[1]
    }
    inter_distances_cpp.update({
        (hash(r), hash(a)): distances['all'][frozenset([hash(r), hash(a)])]
        for r in robots for a in subgoals
    })
    return inter_distances_cpp

def get_subgoals_cpp_for_test(subgoals):
    return [
        mrlsp_accel.SubgoalData(s.prob_feasible,
                                          s.delta_success_cost,
                                          s.exploration_cost,
                                          hash(s)) for s in subgoals
    ]


def test_get_mr_ordering_cost():
    a1 = lsp.core.Frontier(points=np.array([[0], [0]]))
    a2 = lsp.core.Frontier(points=np.array([[1], [1]]))
    a3 = lsp.core.Frontier(points=np.array([[2], [2]]))
    a1.set_props(prob_feasible=0.5, delta_success_cost=10, exploration_cost=10)
    a2.set_props(prob_feasible=0.3, delta_success_cost=20, exploration_cost=20)
    a3.set_props(prob_feasible=0.2, delta_success_cost=30, exploration_cost=30)
    r1 = Robot(1)
    r2 = Robot(2)
    r3 = Robot(3)
    distances = {
        # 'robot': [{a1: 4, a2: 6, a3: 8}, {a1: 6, a2: 4, a3: 6}],
        'goal': {hash(a1): 5, hash(a2): 10, hash(a3): 15},
        'all': {frozenset([hash(a1), hash(a2)]): 2, frozenset([hash(a2), hash(a3)]): 2, frozenset([hash(a1), hash(a3)]): 4,
                frozenset([hash(r1), hash(a1)]): 4, frozenset([hash(r1), hash(a2)]): 6, frozenset([hash(r1), hash(a3)]): 8,
                frozenset([hash(r3), hash(a3)]): 4, frozenset([hash(r3), hash(a2)]): 6, frozenset([hash(r3), hash(a1)]): 8,
                frozenset([hash(r2), hash(a1)]): 6, frozenset([hash(r2), hash(a2)]): 4, frozenset([hash(r2), hash(a3)]): 6,},
    }
    subgoals = [a1, a2, a3]
    subgoals_cpp = get_subgoals_cpp_for_test(subgoals)
    gd_cpp = {hash(s): distances['goal'][hash(s)] for s in subgoals}
    inter_distances_cpp = get_inter_distances_cpp_for_test(subgoals, [r1, r2, r3], distances)

    robots_hash = [hash(r1)]
    calculated_cost_py = get_mr_ordering_cost_py(robots_hash, subgoals, distances)
    calculated_cost_cpp = mrlsp_accel.get_mr_ordering_cost(robots_hash, subgoals_cpp, gd_cpp, inter_distances_cpp)
    actual_cost = (
        14 + a1.prob_feasible * distances['goal'][hash(a1)] + (1 - a1.prob_feasible) * (
            32 + a2.prob_feasible * distances['goal'][hash(a2)] + (1 - a2.prob_feasible) * (
                52 + a3.prob_feasible * distances['goal'][hash(a3)]))
    )
    assert pytest.approx(calculated_cost_py) == actual_cost
    assert pytest.approx(calculated_cost_cpp) == actual_cost

    robots_hash= [hash(r1), hash(r2)]
    calculated_cost_py = get_mr_ordering_cost_py(robots_hash, subgoals, distances)
    calculated_cost_cpp = mrlsp_accel.get_mr_ordering_cost(robots_hash, subgoals_cpp, gd_cpp, inter_distances_cpp)
    actual_cost = (
        14 + a1.prob_feasible * distances['goal'][hash(a1)] + (1 - a1.prob_feasible) * (
            10 + a2.prob_feasible * distances['goal'][hash(a2)] + (1 - a2.prob_feasible) * (
                34 + a3.prob_feasible * distances['goal'][hash(a3)]))
    )
    assert pytest.approx(calculated_cost_py) == actual_cost
    assert pytest.approx(calculated_cost_cpp) == actual_cost

    robots_hash = [hash(r1), hash(r2), hash(r3)]
    calculated_cost_py = get_mr_ordering_cost_py(robots_hash, subgoals, distances)
    calculated_cost_cpp = mrlsp_accel.get_mr_ordering_cost(robots_hash, subgoals_cpp, gd_cpp, inter_distances_cpp)
    actual_cost = (
        14 + a1.prob_feasible * distances['goal'][hash(a1)] + (1 - a1.prob_feasible) * (
            10 + a2.prob_feasible * distances['goal'][hash(a2)] + (1 - a2.prob_feasible) * (
                10 + a3.prob_feasible * distances['goal'][hash(a3)]))
    )
    assert pytest.approx(calculated_cost_py) == actual_cost
    assert pytest.approx(calculated_cost_cpp) == actual_cost


def test_last_robot_finishes_all_subgoals():
    '''Scenario where r2 can reach all the subgoals quickly in least distance'''
    a1 = lsp.core.Frontier(points=np.array([[0], [0]]))
    a2 = lsp.core.Frontier(points=np.array([[1], [1]]))
    a3 = lsp.core.Frontier(points=np.array([[2], [2]]))
    a1.set_props(prob_feasible=0, delta_success_cost=0, exploration_cost=0)
    a2.set_props(prob_feasible=0, delta_success_cost=0, exploration_cost=0)
    a3.set_props(prob_feasible=0, delta_success_cost=0, exploration_cost=0)
    r1 = Robot(1)
    r2 = Robot(2)
    distances = {
        # 'robot': [{a1: 4, a2: 6, a3: 8}, {a1: 6, a2: 4, a3: 6}],
        'goal': {hash(a1): 5, hash(a2): 10, hash(a3): 15},
        'all': {frozenset([hash(a1), hash(a2)]): 11, frozenset([hash(a2), hash(a3)]): 3, frozenset([hash(a1), hash(a3)]): 2,
                frozenset([hash(r1), hash(a1)]): 10, frozenset([hash(r1), hash(a2)]): 3, frozenset([hash(r1), hash(a3)]): 5,
                frozenset([hash(r2), hash(a1)]): 10, frozenset([hash(r2), hash(a2)]): 3, frozenset([hash(r2), hash(a3)]): 5,},
    }
    subgoals = [a1, a2, a3]
    robots_hash = [hash(r1), hash(r2)]
    subgoals_cpp = get_subgoals_cpp_for_test(subgoals)
    gd_cpp = {hash(s): distances['goal'][hash(s)] for s in subgoals}
    inter_distances_cpp = get_inter_distances_cpp_for_test(subgoals, robots_hash, distances)

    calculated_cost_py = get_mr_ordering_cost_py(robots_hash, subgoals, distances)
    calculated_cost_cpp = mrlsp_accel.get_mr_ordering_cost(robots_hash, subgoals_cpp, gd_cpp, inter_distances_cpp)
    assert calculated_cost_py == 8
    assert pytest.approx(calculated_cost_cpp) == 8

def test_last_robot_finishes_other_robot_assignments():
    '''Scenario where r3 can reach last three subgoals. (M Scenario)'''
    a1 = lsp.core.Frontier(points=np.array([[0], [0]]))
    a2 = lsp.core.Frontier(points=np.array([[1], [1]]))
    a3 = lsp.core.Frontier(points=np.array([[2], [2]]))
    a4 = lsp.core.Frontier(points=np.array([[3], [3]]))
    a5 = lsp.core.Frontier(points=np.array([[4], [4]]))
    a1.set_props(prob_feasible=0, delta_success_cost=0, exploration_cost=0)
    a2.set_props(prob_feasible=0, delta_success_cost=0, exploration_cost=0)
    a3.set_props(prob_feasible=0, delta_success_cost=0, exploration_cost=0)
    a4.set_props(prob_feasible=0, delta_success_cost=0, exploration_cost=0)
    a5.set_props(prob_feasible=0, delta_success_cost=0, exploration_cost=0)
    r1 = Robot(1)
    r2 = Robot(2)
    r3 = Robot(3)
    distances = {
        'goal': {hash(a1): 0, hash(a2): 0, hash(a3): 0, hash(a4): 0, hash(a5): 0},
        'all': {frozenset([hash(a1), hash(a2)]): 15, frozenset([hash(a1), hash(a3)]): 16, frozenset([hash(a1), hash(a4)]): 17, frozenset([hash(a1), hash(a5)]): 18,
                frozenset([hash(a2), hash(a3)]): 16.5, frozenset([hash(a2), hash(a4)]): 17.5, frozenset([hash(a2), hash(a5)]): 18.5,
                frozenset([hash(a3), hash(a4)]): 1.1, frozenset([hash(a3), hash(a5)]): 2.3,
                frozenset([hash(a4), hash(a5)]): 1.2,
                frozenset([hash(r1), hash(a1)]): 10, frozenset([hash(r1), hash(a2)]): 11, frozenset([hash(r1), hash(a3)]): 12, frozenset([hash(r1), hash(a4)]): 13.1, frozenset([hash(r1), hash(a5)]): 14.3,
                frozenset([hash(r2), hash(a1)]): 10, frozenset([hash(r2), hash(a2)]): 11, frozenset([hash(r2), hash(a3)]): 12, frozenset([hash(r2), hash(a4)]): 13.1, frozenset([hash(r2), hash(a5)]): 14.3,
                frozenset([hash(r3), hash(a1)]): 10, frozenset([hash(r3), hash(a2)]): 11, frozenset([hash(r3), hash(a3)]): 12, frozenset([hash(r3), hash(a4)]): 13.1, frozenset([hash(r3), hash(a5)]): 14.3,},
    }
    subgoals = [a1, a2, a3, a4, a5]
    robots_hash = [hash(r1), hash(r2), hash(r3)]
    subgoals_cpp = get_subgoals_cpp_for_test(subgoals)
    inter_distances_cpp = get_inter_distances_cpp_for_test(subgoals, robots_hash, distances)
    gd_cpp = {hash(s): distances['goal'][hash(s)] for s in subgoals}

    calculated_cost_py = get_mr_ordering_cost_py(robots_hash, subgoals, distances)
    calculated_cost_cpp = mrlsp_accel.get_mr_ordering_cost(robots_hash, subgoals_cpp, gd_cpp, inter_distances_cpp)
    assert pytest.approx(calculated_cost_py) == 14.3
    assert pytest.approx(calculated_cost_cpp) == 14.3



@pytest.mark.parametrize("num_robots", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("num_frontiers", [1, 2, 3, 4, 5, 6, 7])
def test_best_cost_ordering_random(num_robots, num_frontiers):
    robots = [Robot(i) for i in range(num_robots)]
    robots_hash_dict = {r: hash(r) for r in robots}
    robots_hash = list(robots_hash_dict.values())
    frontiers = [lsp.core.Frontier(points=np.array([[i+1], [i+1]])) for i in range(num_frontiers)]
    hash_to_frontiers = {hash(f): f for f in frontiers}
    for f in frontiers:
        f.set_props(prob_feasible=np.random.rand(),
                    delta_success_cost=np.random.rand() * 15,
                    exploration_cost=np.random.rand() * 30)

    # compute the distances
    distances = {
        'goal': {hash(f): np.random.rand() * 20 for f in frontiers},
        'all': {frozenset([hash(f1), hash(f2)]): np.random.rand() * 10
                for f1 in frontiers + robots
                for f2 in frontiers + robots}
    }

    # brute force solution: fix the first assignments to every permutation of the frontiers (with fewer
    # frontiers than robots, every way of giving each robot a frontier that leaves none unassigned)
    # and complete at lowest cost
    if num_frontiers >= num_robots:
        first_assignments = itertools.permutations(frontiers)
    else:
        first_assignments = [o for o in itertools.product(frontiers, repeat=num_robots)
                             if set(o) == set(frontiers)]
    bf_cost_py = min(get_mr_lowest_cost_state_py(robots_hash, frontiers, distances, ordering=ordering).cost
                     for ordering in first_assignments)

    # solution using our function
    cost_py, ordering_py = get_mr_lowest_cost_ordering_py(robots_hash, frontiers, distances)
    assert pytest.approx(cost_py) == bf_cost_py
    # The ordering holds every assignment, so following it reproduces the cost
    assert len(ordering_py) >= min(num_frontiers, num_robots)
    assert pytest.approx(cost_py) == get_mr_lowest_cost_state_py(
        robots_hash, frontiers, distances, ordering=[hash_to_frontiers[o] for o in ordering_py]).cost

    cost_cpp, ordering_cpp = get_mr_lowest_cost_ordering_cpp(robots_hash, frontiers, distances)
    assert pytest.approx(cost_cpp) == cost_py


def test_one_robot_changes_frontier_mid_way():
    a1 = lsp.core.Frontier(points=np.array([[0], [0]]))
    a2 = lsp.core.Frontier(points=np.array([[1], [1]]))
    a3 = lsp.core.Frontier(points=np.array([[2], [2]]))

    a1.set_props(prob_feasible=0, delta_success_cost=0, exploration_cost=0)
    a2.set_props(prob_feasible=0, delta_success_cost=0, exploration_cost=0)
    a3.set_props(prob_feasible=0, delta_success_cost=0, exploration_cost=0)

    r1 = Robot(1)
    r2 = Robot(2)
    r3 = Robot(3)

    distances = {
        'goal': {hash(a1): 0, hash(a2): 0, hash(a3): 0},
        'all': {frozenset([hash(a1), hash(a2)]): 1, frozenset([hash(a1), hash(a3)]): 2.5, frozenset([hash(a2), hash(a3)]): 3.5,
                frozenset([hash(r1), hash(a1)]): 3, frozenset([hash(r1), hash(a2)]): 4.5, frozenset([hash(r1), hash(a3)]): 3.5,
                frozenset([hash(r2), hash(a1)]): 1.5, frozenset([hash(r2), hash(a2)]): 1, frozenset([hash(r2), hash(a3)]): 4,
                frozenset([hash(r3), hash(a1)]): 8.5, frozenset([hash(r3), hash(a2)]): 9.5, frozenset([hash(r3), hash(a3)]): 6,},
    }
    subgoals = [a1, a2, a3]
    robots_hash = [hash(r1), hash(r2), hash(r3)]
    subgoals_cpp = get_subgoals_cpp_for_test(subgoals)
    gd_cpp = {hash(s): distances['goal'][hash(s)] for s in subgoals}
    inter_distances_cpp = get_inter_distances_cpp_for_test(subgoals, [r1, r2, r3], distances)

    cost = get_mr_ordering_cost_py(robots_hash, subgoals, distances)
    cost_cpp = mrlsp_accel.get_mr_ordering_cost(robots_hash, subgoals_cpp, gd_cpp, inter_distances_cpp)
    assert cost == 4.5
    assert cost_cpp == 4.5


def test_goal_frontier_cost_accumulation():
    a1 = lsp.core.Frontier(points=np.array([[0], [0]]))
    a2 = lsp.core.Frontier(points=np.array([[1], [1]]))
    a3 = lsp.core.Frontier(points=np.array([[2], [2]]))

    a1.set_props(prob_feasible=1.0, delta_success_cost=27, exploration_cost=0)
    a2.set_props(prob_feasible=1.0, delta_success_cost=3, exploration_cost=0)
    a3.set_props(prob_feasible=1.0, delta_success_cost=13, exploration_cost=0)

    r1 = Robot(1)
    r2 = Robot(2)

    distances = {
        'goal': {hash(a1): 61, hash(a2): 73, hash(a3): 137},
        'all': {frozenset([hash(a1), hash(a2)]): 74, frozenset([hash(a1), hash(a3)]): 87, frozenset([hash(a2), hash(a3)]): 77,
                frozenset([hash(r1), hash(a1)]): 38.07, frozenset([hash(r1), hash(a2)]): 43, frozenset([hash(r1), hash(a3)]): 67,
                frozenset([hash(r2), hash(a1)]): 51, frozenset([hash(r2), hash(a2)]): 37.65, frozenset([hash(r2), hash(a3)]): 54,},
    }

    ordering1 = [a1, a2, a3]
    ordering2 = [a1, a3, a2]

    robots_hash = [hash(r1), hash(r2)]
    subgoals1_cpp = get_subgoals_cpp_for_test(ordering1)
    subgoals2_cpp = get_subgoals_cpp_for_test(ordering2)
    gd_cpp = {hash(s): distances['goal'][hash(s)] for s in [a1, a2, a3]}
    inter_distances_cpp = get_inter_distances_cpp_for_test([a1, a2, a3], [r1, r2], distances)

    cost1_py = get_mr_ordering_cost_py(robots_hash, ordering1, distances)
    cost1_cpp = mrlsp_accel.get_mr_ordering_cost(robots_hash, subgoals1_cpp, gd_cpp, inter_distances_cpp)
    cost2_py = get_mr_ordering_cost_py(robots_hash, ordering2, distances)
    cost2_cpp = mrlsp_accel.get_mr_ordering_cost(robots_hash, subgoals2_cpp, gd_cpp, inter_distances_cpp)

    # correct cost ordering 1 = 37.65 + 73 = 110.65 + correction (dsc = 3) = 113.65
    # correct cost ordering 2 = 38.07 + 61 = 99.07 + correction (dsc = 27) = 126.07

    assert cost1_py == 113.65
    assert cost2_py == 126.07
    assert cost1_py == cost1_cpp
    assert cost2_py == cost2_cpp


def test_greedy_endgame_fewer_subgoals_than_robots():
    '''r3 never gets a subgoal from the ordering search, so the end assignment sends it to
    its nearest subgoal (a1: 3 < a2: 4), which r1 already reaches at t=1. Sending r3 to a2
    instead explores a2 at t=4 rather than when r1 or r2 gets there at t=50.'''
    a1 = lsp.core.Frontier(points=np.array([[0], [0]]))
    a2 = lsp.core.Frontier(points=np.array([[1], [1]]))
    a1.set_props(prob_feasible=0.5, delta_success_cost=0, exploration_cost=0)
    a2.set_props(prob_feasible=0.5, delta_success_cost=0, exploration_cost=0)
    r1 = Robot(1)
    r2 = Robot(2)
    r3 = Robot(3)
    distances = {
        'goal': {hash(a1): 0, hash(a2): 0},
        'all': {frozenset([hash(a1), hash(a2)]): 100,
                frozenset([hash(r1), hash(a1)]): 1, frozenset([hash(r1), hash(a2)]): 50,
                frozenset([hash(r2), hash(a1)]): 2, frozenset([hash(r2), hash(a2)]): 50,
                frozenset([hash(r3), hash(a1)]): 3, frozenset([hash(r3), hash(a2)]): 4},
    }
    subgoals = [a1, a2]
    robots_hash = [hash(r1), hash(r2), hash(r3)]

    # r1 -> a1 (t=1), r3 -> a2 (t=4)
    expected_cost = 1 + (1 - a1.prob_feasible) * 3
    cost_py, ordering_py = get_mr_lowest_cost_ordering_py(robots_hash, subgoals, distances)
    assert pytest.approx(cost_py) == expected_cost
    assert ordering_py[2] == hash(a2)
    cost_cpp, ordering_cpp = get_mr_lowest_cost_ordering_cpp(robots_hash, subgoals, distances)
    assert pytest.approx(cost_cpp) == expected_cost
    assert ordering_cpp[2] == hash(a2)


def test_greedy_endgame_free_robot_helps_wrong_subgoal():
    '''r1 finishes a1 at t=1 while r2 (a2, t=10) and r3 (a3, t=100) are still on the way.
    The end assignment sends r1 to its nearest subgoal a2 (5 < 6), duplicating r2, and a3
    is only explored at t=56. Sending r1 to a3 explores it at t=7 and r2 still gets a2 at t=10.'''
    a1 = lsp.core.Frontier(points=np.array([[0], [0]]))
    a2 = lsp.core.Frontier(points=np.array([[1], [1]]))
    a3 = lsp.core.Frontier(points=np.array([[2], [2]]))
    a1.set_props(prob_feasible=0.5, delta_success_cost=0, exploration_cost=0)
    a2.set_props(prob_feasible=0.5, delta_success_cost=0, exploration_cost=0)
    a3.set_props(prob_feasible=0.5, delta_success_cost=0, exploration_cost=0)
    r1 = Robot(1)
    r2 = Robot(2)
    r3 = Robot(3)
    distances = {
        'goal': {hash(a1): 0, hash(a2): 0, hash(a3): 0},
        'all': {frozenset([hash(a1), hash(a2)]): 5, frozenset([hash(a1), hash(a3)]): 6,
                frozenset([hash(a2), hash(a3)]): 50,
                frozenset([hash(r1), hash(a1)]): 1, frozenset([hash(r1), hash(a2)]): 200, frozenset([hash(r1), hash(a3)]): 200,
                frozenset([hash(r2), hash(a1)]): 200, frozenset([hash(r2), hash(a2)]): 10, frozenset([hash(r2), hash(a3)]): 200,
                frozenset([hash(r3), hash(a1)]): 200, frozenset([hash(r3), hash(a2)]): 200, frozenset([hash(r3), hash(a3)]): 100},
    }
    subgoals = [a1, a2, a3]
    robots_hash = [hash(r1), hash(r2), hash(r3)]

    # Greedy: a1 at t=1, a2 at t=6 (r1), a3 at t=56 (r1) -> 1 + 0.5 * 5 + 0.25 * 50 = 16
    # (the greedy search instead returns 14 for ordering [a3, a2, a1], where r2 explores everything)
    # Best:   a1 at t=1, a3 at t=7 (r1), a2 at t=10 (r2) -> 1 + 0.5 * 6 + 0.25 * 3 = 4.75
    expected_cost = 1 + 0.5 * 6 + 0.25 * 3
    cost_py, _ = get_mr_lowest_cost_ordering_py(robots_hash, subgoals, distances)
    assert pytest.approx(cost_py) == expected_cost
    cost_cpp, _ = get_mr_lowest_cost_ordering_cpp(robots_hash, subgoals, distances)
    assert pytest.approx(cost_cpp) == expected_cost
