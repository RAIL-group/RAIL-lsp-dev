import pytest
from mdp import MDP
from pouct_planner import core
from mdp import MDP


def test_deterministic_linear_mdp():
    deterministic_mdp_transitions = {
                'S': {'A': [('S1', 1.0, 10)]},
                'S1': {'B': [('S2', 1.0, 5)]},
                'S2': {}
    }
    state = MDP('S', deterministic_mdp_transitions)
    best_action, cost  = core.po_mcts(state, n_iterations=1000)
    assert best_action == 'A'
    assert cost == 15

def test_deterministic_tree_mdp():
    deterministic_tree_mdp = {
        'S': {'A': [('S1', 1.0, 10)], 'B': [('S2', 1.0, 20)]},
        'S1': {},
        'S2': {}
    }
    state = MDP('S', deterministic_tree_mdp)
    best_action, cost = core.po_mcts(state, n_iterations=1000)
    assert best_action == 'A'
    assert cost == 10

def test_stochastic_mdp_cost():
    stochastic_mdp = {
        'S': {'A': [('S1', 0.8, 5), ('S2', 0.2, 50)],},
        'S1': {'C': [('S3', 1.0, 3)]}, 'S2': {}, 'S3': {}
    }
    state = MDP('S', stochastic_mdp)
    best_action, cost = core.po_mcts(state, n_iterations=10000)
    assert best_action == 'A'
    assert pytest.approx(cost, abs=1.0) == (0.8 * (5 + 3) + 0.2 * 50)

def test_mdp_large_state_space():
    large_state_mdp = {f'S{i}':{f'A': [(f'S{i+1}', 1.0, i)]} for i in range(100)}
    large_state_mdp[f'S100'] = {} # terminal state
    state = MDP('S0', large_state_mdp)
    best_action, cost = core.po_mcts(state, n_iterations=1000)
    assert best_action == 'A'
    assert cost == sum(range(100))

def test_mdp_large_action_space():
    large_action_mdp = {
        'S': {f'A{i}': [('S1', 1.0, i+1)] for i in range(100)},
        'S1': {}
    }
    state = MDP('S', large_action_mdp)
    best_action, cost = core.po_mcts(state, n_iterations=1000)
    assert best_action == 'A0'
    assert cost == 1

@pytest.mark.parametrize('branchings', [3, 4, 5, 6, 7, 8])
# @pytest.mark.parametrize('branchings', [3, 4, 5, 6])
def test_mdp_large_state_action_space(branchings):
    
    dense_mdp = {f'S{i}': {f'A{j}': [(f'S{i+1}', 1.0, (i + j))] for j in range(branchings)} for i in range(branchings)}
    dense_mdp[f'S{branchings}'] = {} # terminal state
    state = MDP('S0', dense_mdp)
    best_action, cost = core.po_mcts(state, n_iterations=50000)
    expected_cost = sum(range(branchings))
    assert pytest.approx(cost, abs=1.0) == expected_cost, f"Cost mismatch: expected {expected_cost}, got {cost}"
    assert best_action == 'A0', f"Best action mismatch: expected 'A0', got {best_action}"

def test_single_action_more_states():
    single_action_mdp = {
        'S': {'A': [('S1', 0.5, 10), ('S2', 0.3, 20), ('S3', 0.2, 30)],
              'B': [('S1', 0.5, 30), ('S2', 0.3, 20), ('S3', 0.2, 10)]},
        'S1': {}, 'S2': {}, 'S3': {}
    }
    state = MDP('S', single_action_mdp)
    best_action, cost = core.po_mcts(state, n_iterations=10000)
    assert best_action == 'A'
    assert pytest.approx(cost, abs=1.0) == (0.5 * 10 + 0.3 * 20 + 0.2 * 30)
