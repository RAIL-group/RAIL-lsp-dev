import pytest
from sctp import base_pomdpstate, base_navigation
from pouct_planner import core
from sctp import graphs


def test_sctpbase_state_lineargraph_cost():
    # Initialize nodes
    start = graphs.Vertex(coords=(0.0, 0.0))
    node1 = graphs.Vertex(coords=(5.0, 0.0))
    goal = graphs.Vertex(coords=(15.0, 0.0))
    nodes = [start, node1, goal]
    assert start.id == 1 and node1.id == 2 and goal.id == 3

    # Make graph and add connectivity
    graph = graphs.Graph(vertices=nodes)
    graph.add_edge(start, node1, 0.0)
    graph.add_edge(node1, goal, 0.0)

    # Why is the position and current_vertex differently initialized?
    robots = graphs.RobotData(robot_id=1, position=(0.0, 0.0), cur_vertex=start.id)
    initial_state = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal.id, robots=robots)

    all_actions = initial_state.get_actions()
    assert len(all_actions) == 1
    action = all_actions[0]
    assert action == node1.id

    best_action, cost = core.po_mcts(initial_state, n_iterations=2000)
    assert best_action == node1.id
    assert cost == pytest.approx(15.0, abs=0.1)


def test_sctpbase_state_disjointgraph_noblock_outcome_states():
    node1 = graphs.Vertex(coords=(0.0, 0.0))
    node2 = graphs.Vertex(coords=(0.0, 4.0))
    node3 = graphs.Vertex(coords=(4.0, 4.0))
    node4 = graphs.Vertex(coords= (5.0, 4.0))
    nodes = [node1, node2, node3, node4]

    graph = graphs.Graph(vertices=nodes)
    graph.add_edge(node1, node2, 0.0)
    graph.add_edge(node3, node4, 0.0)
    graph.add_edge(node1, node4, 0.0)
    graph.add_edge(node2, node3, 0.0)

    robots = graphs.RobotData(robot_id=1, position=(0.0, 0.0), cur_vertex=node1.id)
    initial_state = base_pomdpstate.SCTPBaseState(graph=graph, goal=node3.id, robots=robots)
    all_actions = initial_state.get_actions()

    assert len(all_actions) == 2
    for action in all_actions:
        assert action in [node2.id, node4.id]

    # Lets say we choose node2 action
    action = node2.id
    outcome_states = initial_state.transition(action)
    assert len(outcome_states) == 2
    # One is success history where node2 is not blocked (TRAV)
    success_history = base_pomdpstate.History()
    success_history.add_history(action,
                                robots.cur_vertex,
                                initial_state.edge_probs[tuple(sorted([node1.id, action]))],
                                base_pomdpstate.EventOutcome.TRAV)
    new_state_traversable = base_pomdpstate.get_state_from_history(outcome_states, success_history)
    assert outcome_states[new_state_traversable][0] == 1.0
    assert outcome_states[new_state_traversable][1] == initial_state.edge_costs[tuple(sorted([node1.id, action]))]

    # another is failure history where node2 is blocked (BLOCK)
    failure_history = base_pomdpstate.History()
    failure_history.add_history(action,
                                robots.cur_vertex,
                                initial_state.edge_probs[tuple(sorted([node1.id, action]))],
                                base_pomdpstate.EventOutcome.BLOCK)
    new_state_failure = base_pomdpstate.get_state_from_history(outcome_states, failure_history)
    assert outcome_states[new_state_failure][0] == 0.0
    assert outcome_states[new_state_failure][1] == 10e5


def test_sctpbase_state_lineargraph_with_history():
    # Initialize nodes
    start = graphs.Vertex(coords=(0.0, 0.0))
    node1 = graphs.Vertex(coords=(5.0, 0.0))
    goal = graphs.Vertex(coords=(15.0, 0.0))
    nodes = [start, node1, goal]
    assert start.id == 1 and node1.id == 2 and goal.id == 3

    # Make graph and add connectivity
    graph = graphs.Graph(vertices=nodes)
    graph.add_edge(start, node1, 0.8)
    graph.add_edge(node1, goal, 0.5)

    # Why is the position and current_vertex differently initialized?
    robots = graphs.RobotData(robot_id=1, position=(0.0, 0.0), cur_vertex=start.id)
    node1_blocked_history = base_pomdpstate.History()
    node1_blocked_history.add_history(node1.id, start.id, 0.8, base_pomdpstate.EventOutcome.BLOCK)
    initial_state = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal.id, robots=robots, history=node1_blocked_history)

    all_actions = initial_state.get_actions()
    assert len(all_actions) == 1
    action = node1.id
    # Since node1 is blocked and is in history, we should get only one state in transition where the probability is 1.0
    # and the cost is 10e5
    outcome_states = initial_state.transition(action)

    # WITH HISTORY
    # outcome_states = {node1_blocked_state: (1.0, 10e5)}
    # outcome_states = {node1_blocked_state: (1.0, 10e5), node1_not_blocked_state: (0.0, 4.0)}

    # WITHOUT HISTORY
    # outcome_states = {node1_blocked_state: (0.8, 10e5), node1_not_blocked_state: (0.2, 4.0)}

    print(outcome_states)
    assert len(outcome_states) == 1
    assert outcome_states[list(outcome_states.keys)[0]][0][0] == 1.0
    assert outcome_states[list(outcome_states.keys)[0]][0][1] == 10e5
