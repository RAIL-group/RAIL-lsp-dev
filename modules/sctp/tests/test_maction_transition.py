import pytest
from sctp import sctp_graphs as graphs
from sctp.robot import MCRobot as Robot
from sctp import param, mcstate

def test_mcstate_transition_lgraph():
    print()
    starts, goals, l_graph = graphs.linear_graph_unc()
    ugvs = [Robot(at_node=True, cur_node=starts[0].id) for _ in range(1)]
    
    state = mcstate.MCState(graph=l_graph, goalIDs=[g.id for g in goals], ugvs=ugvs)
    assert state.heuristic == 15.0
    assert state.history.get_data_length() == len(l_graph.vertices)
    assert len(state.state_actions) == len(state.get_actions()) == 1
    assert state.get_actions()[0].sub_targets == [4]
    assert state.get_actions()[0].target == 4
    assert all(ugv.need_action == True for ugv in state.ugvs)

    state2 = state.copy()
    assert state2 != state
    assert state2.state_actions == []
    assert state2.ugvs_actions[0] == state.ugvs_actions[0]

    assert state.get_actions()[0].rtype == param.RobotType.Ground
    state_prob_cost = state.transition(state.get_actions()[0])
    assert len(state_prob_cost) == 2
    state1 = list(state_prob_cost.keys())[0]
    state_b = list(state_prob_cost.keys())[1]
    assert state_b.is_goal_state == True
    state_b.noway2goal = True
    assert state1.action_cost == 2.5
    assert state1.heuristic == 12.5
    assert len(state1.get_actions()) == len(state1.state_actions) == len(state1.ugvs_actions[0]) == 1

    # the second transition - assign action to the ugv then move
    assert state1.get_actions()[0].target == 5
    # print("The next action: ", state1.get_actions()[0])
    state_prob_cost = state1.transition(state1.get_actions()[0])
    assert len(state_prob_cost) == 2
    state2_b = list(state_prob_cost.keys())[1]
    assert state2_b.is_goal_state == True
    state2_p = list(state_prob_cost.keys())[0]
    assert len(state2_p.ugvs_actions[0]) == 1
    assert len(state2_p.get_actions()) == 1
    assert state2_p.ugvs[0].need_action == True
    assert state2_p.action_cost == 7.5
    assert state2_p.heuristic == 5.0
    assert state2_p.noway2goal == False
        
    state_prob_cost = state2_p.transition(state2_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state3_p = list(state_prob_cost.keys())[0]
    assert state3_p.get_actions()[0].rtype == param.RobotType.Ground
    assert state3_p.ugvs[0].need_action == True
    assert state3_p.action_cost == 5.0
    assert state3_p.heuristic == 0.0
    assert state3_p.noway2goal == False 
    assert state3_p.is_goal_state == True
    
def test_mcstate_transition_dg():
    print()
    starts, goals, graph = graphs.disjoint_unc()
    ugvs = [Robot(at_node=True, cur_node=starts[0].id) for _ in range(1)]
    state = mcstate.MCState(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs)
        
    assert state.heuristic == 8.0
    assert state.history.get_data_length() == len(graph.vertices)
    assert len(state.state_actions) == len(state.get_actions()) == 2
    assert all(ugv.need_action == True for ugv in state.ugvs)

    # the first transition
    assert state.get_actions()[0].rtype == param.RobotType.Ground
    assert state.get_actions()[0].target == 5
    assert state.get_actions()[1].target == 8
    state_prob_cost = state.transition(state.get_actions()[0])
    assert len(state_prob_cost) == 2
    state1 = list(state_prob_cost.keys())[0]
    assert state1.action_cost == 2.0
    assert state1.heuristic == 6.0
    assert len(state1.get_actions()) == len(state1.state_actions) == len(state1.ugvs_actions[0]) == 2
    

    # the second transition
    assert state1.get_actions()[1].rtype == param.RobotType.Ground
    assert state1.get_actions()[1].target == 8
    state_prob_cost = state1.transition(state1.get_actions()[1])
    assert len(state_prob_cost) == 2
    state2_b = list(state_prob_cost.keys())[1]
    assert state2_b.is_goal_state == False 
    assert state2_b.noway2goal == False    
    state2_p = list(state_prob_cost.keys())[0]    
    assert len(state2_p.get_actions()) == 2
    assert state2_p.ugvs[0].need_action == True
    assert state2_p.action_cost == pytest.approx(4.83, 0.2)

    # the third transition -
    assert state2_p.get_actions()[0].target == 6
    assert state2_p.get_actions()[1].target == 7
    assert state2_p.get_actions()[0].rtype == param.RobotType.Ground
    state_prob_cost = state2_p.transition(state2_p.get_actions()[0])
    assert len(state_prob_cost) == 2
    state3_p = list(state_prob_cost.keys())[0]    
    assert len(state3_p.get_actions()) == 2
    assert state3_p.ugvs[0].need_action == True
    assert state3_p.action_cost == pytest.approx(5.66, 0.2)

    # the 4th transition 
    assert state3_p.get_actions()[1].rtype == param.RobotType.Ground
    assert state3_p.get_actions()[1].target == 3
    assert state3_p.get_actions()[0].target == 7
    # print("The first action: ", state3_p.get_actions()[0])
    # print("The second action: ", state3_p.get_actions()[1])
    state_prob_cost = state3_p.transition(state3_p.get_actions()[1])
    assert len(state_prob_cost) == 1  
    state4_p = list(state_prob_cost.keys())[0]
    assert len(state4_p.get_actions()) == 1
    assert state4_p.ugvs[0].need_action == True
    assert state4_p.is_goal_state == True 
    assert state4_p.noway2goal == False
    

def test_mcstate_transition_sgraph():
    print()
    starts, goals, graph = graphs.s_graph_2goals()
    ugvs = [Robot(at_node=True, cur_node=starts[0].id) for _ in range(1)]
    state = mcstate.MCState(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs)
    
    assert state.heuristic == pytest.approx(8, 0.05)
    assert state.history.get_data_length() == len(graph.vertices)
    assert len(state.state_actions) == len(state.get_actions()) == 2
    assert all(ugv.need_action == True for ugv in state.ugvs)


#     # the first transition - assign action to the first Ground
#     assert state.get_actions()[0].rtype == param.RobotType.Ground
#     assert state.get_actions()[1].target == 7
#     state_prob_cost = state.transition(state.get_actions()[1])
#     assert len(state_prob_cost) == 1
#     state1 = list(state_prob_cost.keys())[0]
#     assert len(state1.get_actions()) == 2
#     assert state1.get_actions()[0].rtype == param.RobotType.Ground
#     assert state1.action_cost == 0.0
#     assert state1.heuristic == pytest.approx(17.66, 0.05)
#     assert state1.ugvs[0].need_action == False
#     assert state1.ugvs[1].need_action == True
        
#     # the second transition - assign action 7 to the second ugv
#     assert state1.get_actions()[1].target == 7
#     state_prob_cost = state1.transition(state1.get_actions()[1])
#     assert len(state_prob_cost) == 2
#     state2_p = list(state_prob_cost.keys())[0]
#     assert len(state2_p.get_actions()) == 1
#     assert state2_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert state2_p.action_cost == 2.0
#     assert state2_p.heuristic == 16.0
#     assert state2_p.ugvs[0].need_action == True
#     assert state2_p.ugvs[1].need_action == False

#     # the third transition - assign action 3 to ugv 0, then move
#     assert state2_p.get_actions()[0].target == 3
#     assert state2_p.get_actions()[0].robotID == 0
#     state_prob_cost = state2_p.transition(state2_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state3_p = list(state_prob_cost.keys())[0]
#     assert len(state3_p.get_actions()) == 1
#     assert state3_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert state3_p.action_cost == 0.0
#     assert state3_p.ugvs[0].need_action == False
#     assert state3_p.ugvs[1].need_action == True

#     # the 4th transition - assign action 3 to ugv 1, then move
#     assert state3_p.get_actions()[0].target == 3
#     assert state3_p.get_actions()[0].robotID == 1
#     state_prob_cost = state3_p.transition(state3_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state4_p = list(state_prob_cost.keys())[0]
#     assert state4_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert len(state4_p.get_actions()) == len(state4_p.state_actions) == 2
#     assert state4_p.get_actions()[0].robotID == state4_p.get_actions()[0].robotID == 0
#     assert state4_p.action_cost == 2.0
#     assert state4_p.ugvs[0].need_action == True
#     assert state4_p.ugvs[1].need_action == False   

#     # the 5th transition - assign action 8 to ugv 0
#     assert state4_p.get_actions()[0].target == 8
#     assert state4_p.get_actions()[1].target == 11
#     assert state4_p.ugvs_actions[0][0].robotID == state4_p.ugvs_actions[0][1].robotID == 0
#     assert state4_p.ugvs_actions[1][0].robotID == 1
#     assert state4_p.ugvs_actions[1][0].target == 3
#     state4_p.get_actions()[0].robotID == state4_p.get_actions()[1].robotID == 0
#     state_prob_cost = state4_p.transition(state4_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state5_p = list(state_prob_cost.keys())[0]
#     assert state5_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert len(state5_p.get_actions()) == len(state5_p.state_actions) == 2
#     assert state5_p.action_cost == 0.0
#     assert state5_p.ugvs[0].need_action == False
#     assert state5_p.ugvs[1].need_action == True


#     # the 6th transition - assign action 8 to the uav 1
#     assert state5_p.get_actions()[0].target == 8
#     assert state5_p.get_actions()[1].target == 11
#     state_prob_cost = state5_p.transition(state5_p.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state6_p = list(state_prob_cost.keys())[0]
#     assert len(state6_p.get_actions()) == len(state6_p.state_actions) == 1
#     assert state6_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert state6_p.action_cost == 2.0
#     assert state6_p.ugvs[0].need_action == True
#     assert state6_p.ugvs[1].need_action == False    
    
#     # the 7th transition - assign action 2 to ugv 0
#     assert state6_p.get_actions()[0].target == 2
#     state_prob_cost = state6_p.transition(state6_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state7_p = list(state_prob_cost.keys())[0]
#     assert len(state7_p.get_actions()) == len(state7_p.state_actions) == 1
#     assert state7_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert state7_p.action_cost == 0.0
#     assert state7_p.ugvs[0].need_action == False
#     assert state7_p.ugvs[1].need_action == True    
# # The poses of UGV 0 is: [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [4.0, 4.0], [2.586, 2.586]] and current pose [2.586 2.586]
# # The poses of UGV 1 is: [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [4.0, 4.0], [6.0, 4.0]] and current pose [6. 4.]
 

#     # the 8th transition - assign action 2 to UGV 1
#     assert state7_p.get_actions()[0].target == 2
#     state_prob_cost = state7_p.transition(state7_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state8_p = list(state_prob_cost.keys())[0]
#     assert len(state8_p.get_actions()) == len(state8_p.state_actions) == 3
#     assert state8_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert state8_p.action_cost == 2.0
#     assert state8_p.ugvs[0].need_action == True
#     assert state8_p.ugvs[1].need_action == False    

#     # the 9th transition - assign action 6 to UGV 0, then move
#     assert state8_p.get_actions()[0].target == 6
#     state_prob_cost = state8_p.transition(state8_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state9_p = list(state_prob_cost.keys())[0]
#     assert len(state9_p.get_actions()) == len(state9_p.state_actions) == 3
#     assert state9_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert state9_p.action_cost == 0.0
#     assert state9_p.ugvs[0].need_action == False
#     assert state9_p.ugvs[1].need_action == True    

    # print(f"The pose of UGV 0 is: {state9_p.ugvs[0].cur_pose}, last node: {state9_p.ugvs[0].last_node}")
    # print(f"The pose of UGV 1 is: {state9_p.ugvs[1].cur_pose}, last node: {state9_p.ugvs[1].last_node}")

