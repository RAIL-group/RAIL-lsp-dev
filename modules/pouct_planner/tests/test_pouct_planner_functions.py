import pytest
from pouct_planner import core
from mdp import MDP

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
    assert pytest.approx(cost / iter, rel=0.1) == actual_cost

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
    assert pytest.approx(best_action_cost, rel=0.1) == child2_rollout_sim

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
            assert pytest.approx(count[s] / iterations, rel=0.2) == prob
