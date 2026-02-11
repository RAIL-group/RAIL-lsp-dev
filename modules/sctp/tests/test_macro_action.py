import pytest
from sctp import sctp_graphs as graphs
from sctp import mcstate, param
from sctp.param import EventOutcome, RobotType
import numpy as np
import random
from sctp.robot import MCRobot as Robot
from sctp.utils import plotting

def test_mcstate_create_mcactions_lgraph():
    print()
    starts, goals, l_graph = graphs.linear_graph_unc()
    robots = [Robot(cur_node=starts[0].id, at_node=True)]
    # for poi in l_graph.pois:
    #     poi.block_prob = 0.0
    l_graph.pois[0].block_prob = 0.0
    init_state = mcstate.MCState(graph=l_graph, goalIDs=[g.id for g in goals], ugvs=robots)
    assert init_state.pg_adjacency.shape[0] == len(l_graph.vertices)
    assert len(init_state.get_actions()) == 1
    
    print(f"Available macro actions: {init_state.get_actions()[0]}")
    

def test_mcstate_create_mcactions_sgraph():
    print()
    starts, goals, graph = graphs.s_graph_unc()
    robots = [Robot(cur_node=starts[0].id, at_node=True)]
    graph.pois[2].block_prob = 0.0
    graph.pois[1].block_prob = 0.0
    graph.pois[-1].block_prob = 0.0
    init_state = mcstate.MCState(graph=graph, goalIDs=[g.id for g in goals], ugvs=robots)
    assert init_state.pg_adjacency.shape[0] == len(graph.vertices)
    assert len(init_state.get_actions()) == 3
    print(f"Available macro actions: {init_state.get_actions()[0]}")
    print(f"Available macro actions: {init_state.get_actions()[1]}")
    print(f"Available macro actions: {init_state.get_actions()[2]}")
    
    sub_targets1 = [5]
    distances1 = [2.8]
    mcaction1 = mcstate.MAction(start=1, sub_targets=sub_targets1, distances=distances1, robotID=0)
    
    sub_targets2 = [6,3,7,2,5]
    distances2 = [2,2,2,2,2.8]
    mcaction2 = mcstate.MAction(start=2, sub_targets=sub_targets2, distances=distances2, robotID=1)
    assert mcaction1 == mcaction2
    assert init_state.history.get_action_outcome(mcaction1) == param.EventOutcome.CHANCE
    

def test_mcstate_create_mcactions_mgraph():
    print()
    starts, goals, graph = graphs.m_graph_unc()
    robots = [Robot(cur_node=starts[0].id, at_node=True)]
    graph.pois[2].block_prob = 0.0
    graph.pois[1].block_prob = 0.0
    graph.pois[-1].block_prob = 0.0
    init_state = mcstate.MCState(graph=graph, goalIDs=[g.id for g in goals], ugvs=robots)
    assert init_state.pg_adjacency.shape[0] == len(graph.vertices)
    assert len(init_state.get_actions()) == 5
    for i in range(5):
        print(f"Available macro actions: {init_state.get_actions()[i]}")
    
    sub_targets1 = [5]
    # time1 = [2.8]
    mcaction1 = mcstate.MAction(start=1, sub_targets=sub_targets1)
    
    sub_targets2 = [6,3,7,2,5]
    # time2 = [2,2,2,2,2.8]
    mcaction2 = mcstate.MAction(start=1, sub_targets=sub_targets2)
    assert mcaction1 == mcaction2
    assert init_state.history.get_action_outcome(mcaction1) == param.EventOutcome.TRAV
    
