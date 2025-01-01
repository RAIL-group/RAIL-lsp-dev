from mdp import MDP
from pouct_planner import core
from mdp import MDP


def test_deterministic_mdp():
    deterministic_mdp_transitions = {
                'S': {'A': [('S1', 1.0, 10)]},
                'S1': {'B': [('S2', 1.0, 5)]},
                'S2': {}
    }
    state = MDP('S', deterministic_mdp_transitions)

    best_action, cost  = core.po_mcts(state, n_iterations=1000)
    assert best_action == 'A'
    assert cost == 15
