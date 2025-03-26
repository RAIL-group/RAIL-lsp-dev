import pytest
from pouct_planner import core
from .mdp import MDP

def mdp(current_state='S'):
    mdp_transitions = {
        'S': {
            'A': [('S1', 0.3, 20), ('S2', 0.7, 80)],
            'B': [('S3', 0.8, 50), ('S4', 0.2, 90)]
        },
        'S2': {
            'B': [('S21', 0.8, 10), ('S22', 0.2, 30)],
        },
        'S4': {
            'A': [('S41', 0.3, 40), ('S42', 0.7, 60)]
        },
        'S1': {}, 'S3': {}, 'S21': {}, 'S22': {}, 'S41': {}, 'S42': {}
    }
    return MDP(current_state, mdp_transitions)

################################################################
# Tests for POUCTNode class functions
################################################################
def test_is_fully_expanded():
    not_expanded_node = core.POUCTNode(mdp())
    assert not not_expanded_node.is_fully_expanded()

    while not not_expanded_node.is_fully_expanded():
        _ = not_expanded_node.unexplored_actions.pop()
    assert not_expanded_node.is_fully_expanded()

    expanded_node = core.POUCTNode(mdp('S1'))
    assert expanded_node.is_fully_expanded()

def test_is_terminal_node():
    non_terminal_node = core.POUCTNode(mdp())
    assert not non_terminal_node.is_terminal_node()

    terminal_node = core.POUCTNode(mdp('S1'))
    assert terminal_node.is_terminal_node()

################################################################
# Tests for rollout function
################################################################
def test_rollout_fn():
    node = core.POUCTNode(mdp())
    cost = core.rollout(node)
    assert cost > 0

def test_rollout_expected_cost():
    node = core.POUCTNode(mdp())
    iter = 5000
    cost = 0
    for _ in range(iter):
        cost += core.rollout(node)

    actual_cost = 0.5 * (0.3 * 20 + 0.7 * (0.8 * (80 + 10) + 0.2 * (80 + 30))) + \
                  0.5 * (0.8 * 50 + 0.2 * (0.3 * (90 + 40) + 0.7 * (90 + 60)))
    assert pytest.approx(cost / iter, abs=2.0) == actual_cost

def test_rollout_values_within_bounds():
    mdp_transitions = {
        'S': {'A': [('S1', 0.4, 10), ('S2', 0.6, 20)], 'B': [('S3', 0.5, 30), ('S4', 0.5, 40)]},
        'S1': {'C': [('S5', 0.7, 50), ('S6', 0.3, 60)]},
        'S2': {'D': [('S7', 0.5, 70), ('S8', 0.5, 80)]},
        'S3': {'E': [('S9', 0.8, 90), ('S10', 0.2, 100)]},
        'S4': {'F': [('S9', 0.6, 110), ('S10', 0.4, 120)]},
        'S5': {'G': [('S11', 0.9, 130), ('S12', 0.1, 140)]},
        'S6': {'H': [('S11', 0.6, 150), ('S12', 0.4, 160)]},
        'S7': {'I': [('S13', 1.0, 170)]},
        'S8': {'J': [('S13', 1.0, 180)]},
        'S9': {'K': [('S14', 1.0, 190)]},
        'S10': {'L': [('S14', 1.0, 200)]},
        'S11': {},
        'S12': {},
        'S13': {},
        'S14': {}  # Terminal states
    }
    state = MDP('S', mdp_transitions)
    node = core.POUCTNode(state)
    # Min: S -> A -> S1 -> C -> S5 -> G -> S11
    min_rollout_value = 10 + 50 + 130
    # Max: S -> B -> S4 -> F -> S10 -> L -> S14
    max_rollout_value = 40 + 120 + 200

    num_rollouts = 100
    for _ in range(num_rollouts):
        cost = core.rollout(node)
        assert min_rollout_value <= cost <= max_rollout_value

def test_rollout_values_within_bounds_large_state_space():
    branchings = 10
    dense_mdp = {f'S{i}': {f'A{j}': [(f'S{i+1}', 1.0, (i + j))]
                           for j in range(branchings)} for i in range(branchings)}
    dense_mdp[f'S{branchings}'] = {}  # terminal state
    state = MDP('S0', dense_mdp)
    node = core.POUCTNode(state)
    results = [core.rollout(node) for _ in range(100)]
    expected_max_cost = sum(branchings - 1 + i for i in range(branchings))
    assert min(results) >= 0
    assert max(results) <= expected_max_cost

################################################################
# Tests for backpropagate function
################################################################
def test_backpropagate():
    def deterministic_rollout_fn(*args):
        return 100
    root = core.POUCTNode(mdp())
    prev_action = 'A'
    child = core.POUCTNode(mdp('S1'),
                          action=prev_action,
                          parent=root,
                          cost=10)
    sim_result = core.rollout(child, deterministic_rollout_fn)

    core.backpropagate(child, sim_result)
    assert root.action_n[prev_action] == 1
    assert root.action_values[prev_action] == sim_result

    iter = 10
    for _ in range(iter):
        core.backpropagate(child, sim_result)
    assert root.action_n[prev_action] == iter + 1
    assert root.action_values[prev_action] == (iter + 1) * sim_result


################################################################
# Tests for get_best_action and get_best_uct_action function
################################################################
def test_get_best_actions():
    root = core.POUCTNode(mdp())
    child1 = core.POUCTNode(mdp('S1'), parent=root, action='A', cost=10)
    child2 = core.POUCTNode(mdp('S3'), parent=root, action='B', cost=20)
    root.children = {child1, child2}
    n = 20
    child2_rollout_sim = 50
    child1_rollout_sim = 100
    for _ in range(n):
        core.backpropagate(child1, child1_rollout_sim)
        core.backpropagate(child2, child2_rollout_sim)

    best_action = root.get_best_uct_action(C=1.0)
    assert best_action == 'B'

    best_action, best_action_cost = core.get_best_action(root)
    assert best_action == 'B'
    assert pytest.approx(best_action_cost, abs=0.1) == child2_rollout_sim


################################################################
# Tests for traverse function
################################################################
def test_traverse_terminal_node():
    terminal_node = core.POUCTNode(mdp('S1'))
    child = core.traverse(terminal_node)

    assert child == terminal_node
    assert child.is_terminal_node()

def test_traverse_fully_expanded():
    root = core.POUCTNode(mdp())
    while not root.is_fully_expanded():
        _ = core.traverse(root)

    assert root.is_fully_expanded()
    for _ in range(10):
        child = core.traverse(root)
        assert child.parent is not None
        assert child.prev_action in root.action_n


################################################################
# Tests for get_chance_node function
################################################################
def test_chance_node_probabilities():
    '''Test to confirm whether the get_chance_node() function samples next
    state according to the probability distribution of the transition'''
    state = mdp()
    root = core.POUCTNode(state)
    actions = state.get_actions()
    for action in actions:
        actual_transition = {s: prob for s, (prob, _) in state.transition(action).items()}
        count = {s: 0 for s in actual_transition.keys()}
        iterations = 2000
        for _ in range(iterations):
            node = core.get_chance_node(root, action)
            count[node.state] += 1
        for s, prob in actual_transition.items():
            assert pytest.approx(count[s] / iterations, abs=0.2) == prob


################################################################
# Tests for get_best_path function
################################################################
def test_get_best_path():
    root = core.POUCTNode(mdp())
    root.total_n = 10
    child1 = core.POUCTNode(mdp('S1'), parent=root, action='A', cost=20)
    child2 = core.POUCTNode(mdp('S2'), parent=root, action='A', cost=80)
    root.action_outcomes = {'A': {child1: 0.3, child2: 0.7}}
    root.action_n = {'A': 10}
    root.action_values = {'A': 0} # Cost doesn't matter
    root.children = {child1, child2}
    child1.total_n = 2
    child2.total_n = 8 # child 2 is explored more

    child21 = core.POUCTNode(mdp('S21'), parent=child1, action='B', cost=10)
    child22 = core.POUCTNode(mdp('S22'), parent=child1, action='B', cost=30)
    child2.action_outcomes = {'B': {child21: 0.8, child22: 0.2}}
    child2.action_n = {'B': 8}
    child2.action_values = {'B': 0}
    child2.children = {child21, child22}

    path, cost = core.get_best_path(root)
    actual_path = ['A', 'B']
    assert path == actual_path
