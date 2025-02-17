import pytest
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.param import EventOutcome

def test_sctp_actions():
    start_node = graphs.Vertex(coord=(0.0, 0.0))
    goal_node = graphs.Vertex(coord=(15.0, 0.0))
    node1 = graphs.Vertex(coord=(5.0, 0.0))
    
    graph = graphs.Graph(vertices=[start_node, goal_node, node1])
    graph.add_edge(start_node, node1, 0.0)
    graph.add_edge(node1, goal_node, 0.0)
    action1 = core.TeamAction(start=start_node.id, end=node1.id)
    action2 = core.TeamAction(start=node1.id, end=goal_node.id)
    assert action1 != action2
    action3 = core.TeamAction(start=start_node.id, end=node1.id)
    assert action1 == action3

def test_sctp_metric():
    metric1 = core.sctp_metric(10.0, 5.0)
    metric2 = core.sctp_metric(5.0, 10.0)
    assert metric1 > metric2
    metric3 = core.sctp_metric(10.0, 10.0)
    assert metric1 < metric3
    assert metric2 < metric3
    metric4 = core.sctp_metric(10.0, 5.0)
    assert metric1 == metric4
    assert metric3 != metric4
    assert metric1 + metric2 == core.sctp_metric(15.0, 15.0)

def test_sctp_function_init():
    start, goal, l_graph, robots = graphs.linear_graph_unc()

    action1 = core.TeamAction(start=start.id, end=start.neighbors[0])
    state = core.SCTPState(graph=l_graph, goal=goal.id, robots=robots)
#     edge = base_pomdpstate.get_edge_from_action(state, action1)
#     assert edge == graph.edges[0]
#     action1 = base_pomdpstate.Action(start_node=node1.id, target_node=goal_node.id)
#     edge2 = base_pomdpstate.get_edge_from_action(state, action1)
#     assert edge2 == graph.edges[1]

# def test_sctpbase_function_rollout():
#     start, goal, l_graph, robots = graphs.s_graph_unc()
#     # start, goal, l_graph, robots = graphs.m_graph_unc()
#     init_state = base_pomdpstate.SCTPBaseState(graph=l_graph, goal=goal.id, robots=robots)
#     all_actions = init_state.get_actions()
#     assert len(all_actions) == 2

#     action2 = base_pomdpstate.Action(start_node=start.id, target_node=2)
#     action3 = base_pomdpstate.Action(start_node=start.id, target_node=3)
#     for action in all_actions:
#         assert action in [action2, action3]


#     best_action, cost, path = core.po_mcts(init_state, n_iterations=10,
#                                 C=15.0, rollout_fn=base_pomdpstate.sctpbase_rollout)
