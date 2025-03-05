import pytest
from sctp import sctp_graphs as graphs
from sctp import core
from pouct_planner import core as planner
from sctp.param import EventOutcome

def test_sctp_planner_lg():
    exp_param=100.0
    start, goal, l_graph, robots = graphs.linear_graph_unc()
    init_state = core.SCTPState(graph=l_graph, goal=goal.id, robots=robots)
    # state_actions = init_state.get_actions()
    # ba, ec, pc  = planner.po_mcts(init_state, C=exp_param, n_iterations=1000)
    ba, ec, pc  = planner.po_mcts(init_state, C=exp_param, n_iterations=100,\
                                                rollout_fn=core.sctp_rollout)
    # assert ba.target == 4
    for p in pc[0]:
        print(p)
    # path_cost[0]
    