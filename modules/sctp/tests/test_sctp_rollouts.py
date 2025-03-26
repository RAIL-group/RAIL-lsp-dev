import pytest
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.robot import Robot
from sctp.param import RobotType, VEL_RATIO
from sctp.param import EventOutcome
from pouct_planner import core as pomcp
  
def test_sctp_rollout_integrating_dgraph():
    start, goal, d_graph, robots = graphs.disjoint_unc()
    init_state = core.SCTPState(graph=d_graph, goal=goal.id, robots=robots) 
    cost = core.sctp_rollout(init_state)

def test_sctp_rollout_integrating_sgraph():
    start, goal, s_graph, robots = graphs.s_graph_unc()
    init_state = core.SCTPState(graph=s_graph, goal=goal.id, robots=robots)
    cost = core.sctp_rollout(init_state)

def test_sctp_rollout2_sgraph():
    start, goal, s_graph, robots = graphs.s_graph_unc()
    init_state = core.SCTPState(graph=s_graph, goal=goal.id, robots=robots)
    node = pomcp.POUCTNode(init_state)
    print(f"Position of robot {node.state.robot.cur_pose}")
    for i in range(500):
        cost = core.sctp_rollout2(node)
    

def test_sctp_rollout_integrating_mgraph():
    start, goal, s_graph, robots = graphs.m_graph_unc()
    init_state = core.SCTPState(graph=s_graph, goal=goal.id, robots=robots) 
    cost = core.sctp_rollout(init_state)


def test_sctp_rollout2_mgraph():
    start, goal, m_graph, robots = graphs.m_graph_unc()
    init_state = core.SCTPState(graph=m_graph, goal=goal.id, robots=robots)
    node = pomcp.POUCTNode(init_state)
    print(f"Position of robot {node.state.robot.cur_pose}")
    for i in range(1000):
        cost = core.sctp_rollout2(node)
    