import pytest
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.robot import Robot
from sctp.param import RobotType, VEL_RATIO
from sctp.param import EventOutcome
  
def test_sctp_rollout_integrating():
    start, goal, l_graph, robots = graphs.disjoint_unc()
    init_state = core.SCTPState(graph=l_graph, goal=goal.id, robots=robots)
    # init_state.history.add_history(action=core.Action(target=7), outcome=EventOutcome.BLOCK)
    # init_state.history.add_history(action=core.Action(target=5), outcome=EventOutcome.TRAV)
    # 
    cost = core.sctp_rollout(init_state)