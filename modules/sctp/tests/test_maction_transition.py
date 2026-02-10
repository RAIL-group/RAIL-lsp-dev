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
    
    # the first transition - assign action to the drone
    assert state.get_actions()[0].rtype == param.RobotType.Ground
    state_prob_cost = state.transition(state.get_actions()[0])
    assert len(state_prob_cost) == 2
    state1 = list(state_prob_cost.keys())[0]
    assert state1.action_cost == 2.5
    assert state1.heuristic == 12.5
    assert len(state1.get_actions()) == len(state1.state_actions) == len(state1.ugvs_actions[0]) == 1

    # the second transition - assign action to the ugv then move
    # assert state1.get_actions()[0].target == 4
    # state_prob_cost = state1.transition(state1.get_actions()[0])
    # assert state1.get_actions()[0].rtype == param.RobotType.Ground
    # assert len(state_prob_cost) == 2
    # state2_p = list(state_prob_cost.keys())[0]
    # assert state2_p.get_actions()[0].rtype == param.RobotType.Drone
    # assert len(state2_p.ugvs_actions[0]) == 2
    # assert len(state2_p.get_actions()) == 1
    # assert state2_p.ugvs[0].need_action == True
    # assert state2_p.uavs[0].need_action == True
    # assert state2_p.action_cost == 2.5/param.VEL_RATIO
    # assert state2_p.heuristic == pytest.approx(15.0-state2_p.action_cost, 0.05)
    # assert state2_p.noway2goal == False
    
    # state2_b = list(state_prob_cost.keys())[1]
    # assert state2_b.ugvs[0].need_action == True
    # assert state2_b.uavs[0].need_action == True
    # assert state2_b.action_cost == 2.5/param.VEL_RATIO
    # assert state2_b.heuristic == param.NOWAY_PEN
    # assert state2_b.noway2goal == True
    # assert state2_b.is_goal_state == True
    
    # # the third transition - assign action to the uav 
    # state_prob_cost = state2_p.transition(state2_p.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state3_p = list(state_prob_cost.keys())[0]
    # assert state3_p.get_actions()[0].rtype == param.RobotType.Ground
    # assert state3_p.ugvs[0].need_action == True
    # assert state3_p.uavs[0].need_action == False
    # assert state3_p.action_cost == 0.0
    
    # # the fourth transition - assign action 1 to the ugv then move 
    # state_prob_cost = state3_p.transition(state3_p.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state4_p = list(state_prob_cost.keys())[0]
    # assert state4_p.get_actions()[0].rtype == param.RobotType.Ground
    # assert state4_p.ugvs[0].need_action == True
    # assert state4_p.uavs[0].need_action == False
    # assert state4_p.action_cost == (2.5/param.VEL_RATIO)
    
    # # action 4
    # state_prob_cost = state3_p.transition(state3_p.get_actions()[1])
    # assert len(state_prob_cost) == 1
    # state4_p = list(state_prob_cost.keys())[0]
    # assert state4_p.get_actions()[0].rtype == param.RobotType.Ground
    # assert state4_p.ugvs[0].need_action == True
    # assert state4_p.uavs[0].need_action == False
    # assert state4_p.action_cost == (2.5-2.5/param.VEL_RATIO)
    
    # # the fifth transition - assign action 2 to the ugv, then move
    # # then uav 0 reach node v5 first/ reset action both ugvs and uavs 
    # assert state4_p.get_actions()[0].target == 2
    # state_prob_cost = state4_p.transition(state4_p.get_actions()[0])
    # assert len(state_prob_cost) == 2

    # # blocked state
    # state5_b = list(state_prob_cost.keys())[1]
    # assert state5_b.get_actions()[0].rtype == param.RobotType.Drone
    # assert state5_b.ugvs[0].need_action == True
    # assert state5_b.uavs[0].need_action == True
    # assert state5_b.action_cost == pytest.approx(2.5/param.VEL_RATIO, 0.05)
    # assert state5_b.uavs[0].last_node == 5
    # assert state5_b.noway2goal == True 
    # assert state5_b.is_goal_state == True
    

    # state5_p = list(state_prob_cost.keys())[0]
    # assert state5_p.get_actions()[0].rtype == param.RobotType.Drone
    # assert state5_p.ugvs[0].need_action == True
    # assert state5_p.uavs[0].need_action == True
    # assert state5_p.action_cost == pytest.approx(2.5/param.VEL_RATIO, 0.05)
    # assert state5_p.uavs[0].last_node == 5
    
    # # the sixth transition - assign action 3 to the uav
    # assert state5_p.get_actions()[0].target == 3
    # state_prob_cost = state5_p.transition(state5_p.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state6_p = list(state_prob_cost.keys())[0]
    # assert state6_p.get_actions()[0].rtype == param.RobotType.Ground
    # assert state6_p.ugvs[0].need_action == True
    # assert state6_p.uavs[0].need_action == False
    # assert state6_p.action_cost == 0.0
    # assert state6_p.uavs[0].last_node == 5
    # assert len(state6_p.get_actions()) == 2
    # assert state6_p.get_actions()[1].target == 2
    
    # # the 7th transition - assign action 2 to the ugv then move
    # # both teams ugv and uav reach their targets
    # state_prob_cost = state6_p.transition(state6_p.get_actions()[1])
    # assert len(state_prob_cost) == 1
    # state7_p = list(state_prob_cost.keys())[0]
    # assert state7_p.ugvs[0].last_node == 2
    # assert state7_p.ugvs[0].need_action == False
    # assert state7_p.uavs[0].need_action == True
    # assert state7_p.action_cost == pytest.approx(5.0/param.VEL_RATIO, 0.05)
    # assert state7_p.get_actions()[0].rtype == param.RobotType.Drone
    
    # # the 8th transition - assign wait action for the drone
    # state_prob_cost = state7_p.transition(state7_p.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state8_p = list(state_prob_cost.keys())[0]
    # assert len(state8_p.get_actions()) == 1
    # assert state8_p.ugvs[0].last_node == 2
    # assert state8_p.ugvs[0].need_action == True
    # assert state8_p.uavs[0].need_action == False
    # assert state8_p.action_cost == pytest.approx(0, param.APPROX_TIME)
    # assert state8_p.get_actions()[0].rtype == param.RobotType.Ground
    
    # # the 9th transition - assign action 5 to the ugv then move
    # state_prob_cost = state8_p.transition(state8_p.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state9_p = list(state_prob_cost.keys())[0]
    # # print(f"The position of ugv: {state9_p.ugvs[0].cur_pose}, last node: {state9_p.ugvs[0].last_node}")
    # assert len(state9_p.get_actions()) == 1
    # assert state9_p.ugvs[0].last_node == 5
    # assert state9_p.uavs[0].last_node == 3
    # assert state9_p.ugvs[0].need_action == True
    # assert state9_p.uavs[0].need_action == False
    # assert state9_p.action_cost == pytest.approx(5.0, param.APPROX_TIME)
    # assert state9_p.get_actions()[0].rtype == param.RobotType.Ground
    
    # # the 10th transition - assign action 3 to the ugv then move
    # state_prob_cost = state9_p.transition(state9_p.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state10_p = list(state_prob_cost.keys())[0]
    # print(f"The position of ugv: {state10_p.ugvs[0].cur_pose}, last node: {state10_p.ugvs[0].last_node}")
    # assert len(state10_p.get_actions()) == 1
    # assert state10_p.ugvs[0].last_node == 3
    # assert state10_p.uavs[0].last_node == 3
    # assert state10_p.ugvs[0].need_action == True
    # assert state10_p.uavs[0].need_action == False
    # assert state10_p.action_cost == pytest.approx(5.0, param.APPROX_TIME)
    # assert state10_p.is_goal_state == True
    # assert state10_p.noway2goal == False

# def test_jsap_transition_1uav1ugv_dg():
#     print()
#     starts, goals, graph = graphs.disjoint_unc()
#     uavs = [Robot(position=[0.0, 0.0], cur_node=starts[0].id, at_node=True, robot_type=param.RobotType.Drone) \
#                     for _ in range(1)]
#     ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[0].id, at_node=True) for _ in range(1)]

#     state = jsap.JSAPState(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=uavs,
#                            n_maps=80, useAVP=False, max_uanum=1)
    
#     assert state.heuristic == 8.0
#     assert state.history.get_data_length() == len(graph.vertices)
#     assert len(state.state_actions) == len(state.get_actions()) == 4
#     assert all(uav.need_action == True for uav in state.uavs)
#     assert all(ugv.need_action == True for ugv in state.ugvs)
#     assert len(state.assigned_pois) == 0

#     # the first transition - assign action 5 to the drone
#     assert len(state.get_actions()) == 4
#     assert state.get_actions()[0].rtype == param.RobotType.Drone
#     assert state.get_actions()[0].target == 5
#     state_prob_cost = state.transition(state.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state1 = list(state_prob_cost.keys())[0]
#     assert state1.action_cost == 0.0
#     assert state1.heuristic == 8.0
#     assert len(state1.get_actions()) == len(state1.state_actions) == len(state1.ugvs_actions[0]) == 2

#     # the second transition - assign action 8 to the ugv, then move
#     assert state1.get_actions()[1].rtype == param.RobotType.Ground
#     assert state1.get_actions()[1].target == 8
#     state_prob_cost = state1.transition(state1.get_actions()[1])
#     assert len(state_prob_cost) == 2
#     state2_p = list(state_prob_cost.keys())[0]    
#     assert len(state2_p.get_actions()) == 3
#     assert state2_p.ugvs[0].need_action == True
#     assert state2_p.uavs[0].need_action == True
#     assert state2_p.action_cost == 2.0/param.VEL_RATIO

#     # the third transition - assign action 6 to the uav
#     assert state2_p.get_actions()[0].target == 6
#     assert state2_p.get_actions()[0].rtype == param.RobotType.Drone
#     state_prob_cost = state2_p.transition(state2_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state3_p = list(state_prob_cost.keys())[0]    
#     assert len(state3_p.get_actions()) == 2
#     assert state3_p.ugvs[0].need_action == True
#     assert state3_p.uavs[0].need_action == False
#     assert state3_p.action_cost == 0.0

#     # the 4th transition - assign action 8 to the ugv and move
#     assert state3_p.get_actions()[1].rtype == param.RobotType.Ground
#     assert state3_p.get_actions()[1].target == 8
#     state_prob_cost = state3_p.transition(state3_p.get_actions()[1])
#     assert len(state_prob_cost) == 2
#     state4_p = list(state_prob_cost.keys())[0]
#     assert state4_p.get_actions()[0].rtype == param.RobotType.Drone
#     assert len(state4_p.get_actions()) == 2
#     assert state4_p.ugvs[0].need_action == True
#     assert state4_p.uavs[0].need_action == True
#     assert state4_p.action_cost == pytest.approx(1.491, 0.05)

#     # the 5th transition - assign action 7 to the uav
#     assert state4_p.get_actions()[0].target == 7
#     state_prob_cost = state4_p.transition(state4_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state5_p = list(state_prob_cost.keys())[0]
#     assert state5_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert len(state5_p.get_actions()) == 2
#     assert state5_p.ugvs[0].need_action == True
#     assert state5_p.uavs[0].need_action == False
#     assert state5_p.action_cost == 0.0
    
#     # the 6th transition - assign action 8 to the ugv then move, the drone reach goal first
#     assert state5_p.get_actions()[1].target == 8
#     state_prob_cost = state5_p.transition(state5_p.get_actions()[1])
#     assert len(state_prob_cost) == 2
#     state6_p = list(state_prob_cost.keys())[0]
#     # print(f"The position of ugv: {state6_p.ugvs[0].cur_pose}, last node: {state6_p.ugvs[0].last_node}")
#     # print(f"The position of drone: {state6_p.uavs[0].cur_pose}, last node: {state6_p.uavs[0].last_node}")
#     assert state6_p.get_actions()[0].rtype == param.RobotType.Drone
#     assert len(state6_p.get_actions()) == 1
#     assert state6_p.ugvs[0].need_action == True
#     assert state6_p.uavs[0].need_action == True
#     assert state6_p.action_cost == 2.0/param.VEL_RATIO

#     # the 7th transition - assign action 8 to the uav 
#     assert state6_p.get_actions()[0].target == 8
#     state_prob_cost = state6_p.transition(state6_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state7_p = list(state_prob_cost.keys())[0]
#     assert state7_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert len(state7_p.get_actions()) == 2
#     assert state7_p.ugvs[0].need_action == True
#     assert state7_p.uavs[0].need_action == False

#     # the 8th transition - assign action to the ugv, then move 
#     assert state7_p.get_actions()[1].target == 8
#     state_prob_cost = state7_p.transition(state7_p.get_actions()[1])
#     assert len(state_prob_cost) == 2
#     state8_p = list(state_prob_cost.keys())[0]
#     assert state8_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert len(state8_p.get_actions()) == 1
#     assert state8_p.ugvs[0].need_action == True
#     assert state8_p.uavs[0].need_action == True
#     assert state8_p.action_cost == pytest.approx(0.005, abs=0.001)

#     # the 9th transition - assign action 4 to the ugv 
#     assert state8_p.get_actions()[0].target == 4
#     state_prob_cost = state8_p.transition(state8_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state9_p = list(state_prob_cost.keys())[0]
#     assert state9_p.get_actions()[0].rtype == param.RobotType.Drone
#     assert len(state9_p.get_actions()) == 1
#     assert state9_p.ugvs[0].need_action == False
#     assert state9_p.uavs[0].need_action == True
#     assert state9_p.action_cost == 0.0

#     # the 10th transition - assign action 3 (goal) to the uav, then move
#     assert state9_p.get_actions()[0].target == 3
#     state_prob_cost = state9_p.transition(state9_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state10_p = list(state_prob_cost.keys())[0]
#     assert state10_p.get_actions()[0].rtype == param.RobotType.Drone
#     assert len(state10_p.get_actions()) == 1
#     assert state10_p.ugvs[0].need_action == False
#     assert state10_p.uavs[0].need_action == True
    
#     # the 11th transition - assign action to the uav, (staying at goal) then move ugv
#     assert state10_p.get_actions()[0].target == 3
#     state_prob_cost = state10_p.transition(state10_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state11_p = list(state_prob_cost.keys())[0]
#     assert state11_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert len(state11_p.get_actions()) == 1
#     assert state11_p.ugvs[0].need_action == True
#     assert state11_p.uavs[0].need_action == False

#     # the 12th transition - assign action to the ugv, then move
#     assert state11_p.get_actions()[0].target == 6
#     state_prob_cost = state11_p.transition(state11_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state12_p = list(state_prob_cost.keys())[0]
#     assert state12_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert len(state12_p.get_actions()) == 1
#     assert state12_p.ugvs[0].need_action == True
#     assert state12_p.uavs[0].need_action == False
#     assert state12_p.action_cost == pytest.approx(2.828, 0.05)

#     # the 13th transition - assign action to the ugv, then move
#     assert state12_p.get_actions()[0].target == 3
#     state_prob_cost = state12_p.transition(state12_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state13_p = list(state_prob_cost.keys())[0]
#     assert state13_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert len(state13_p.get_actions()) == 1
#     assert state13_p.ugvs[0].need_action == True
#     assert state13_p.uavs[0].need_action == False
#     assert state13_p.action_cost == pytest.approx(2.828, 0.05)
#     assert state13_p.is_goal_state == True
#     assert state13_p.noway2goal == False
    
# def test_jsap_transition_dg_finish_sametime():
#     print()
#     starts, goals, graph = graphs.disjoint_unc()
#     uavs = [Robot(position=[0.0, 0.0], cur_node=starts[0].id, at_node=True, robot_type=param.RobotType.Drone) \
#                     for _ in range(1)]
#     ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[0].id, at_node=True) for _ in range(1)]

#     state = jsap.JSAPState(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=uavs,
#                            n_maps=80, useAVP=False, max_uanum=1)

#     # the first transition - assign action 5 to the drone
#     assert len(state.get_actions()) == 4
#     assert state.get_actions()[0].rtype == param.RobotType.Drone
#     assert state.get_actions()[0].target == 5
#     state_prob_cost = state.transition(state.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state1 = list(state_prob_cost.keys())[0]
#     assert state1.action_cost == 0.0
#     assert state1.heuristic == 8.0
#     assert len(state1.get_actions()) == len(state1.state_actions) == len(state1.ugvs_actions[0]) == 2

#     # the second transition - assign action 8 to the ugv, then move
#     assert state1.get_actions()[0].rtype == param.RobotType.Ground
#     assert state1.get_actions()[0].target == 5
#     state_prob_cost = state1.transition(state1.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state2_p = list(state_prob_cost.keys())[0]    
#     assert len(state2_p.get_actions()) == 3
#     assert state2_p.ugvs[0].need_action == True
#     assert state2_p.uavs[0].need_action == True
#     assert state2_p.action_cost == 2.0/param.VEL_RATIO

#     # the third transition - assign action 6 to the uav
#     assert state2_p.get_actions()[1].target == 7
#     assert state2_p.get_actions()[1].rtype == param.RobotType.Drone
#     state_prob_cost = state2_p.transition(state2_p.get_actions()[1])
#     assert len(state_prob_cost) == 1
#     state3_p = list(state_prob_cost.keys())[0]    
#     assert len(state3_p.get_actions()) == 2
#     assert state3_p.ugvs[0].need_action == True
#     assert state3_p.uavs[0].need_action == False
#     assert state3_p.action_cost == 0.0

#     # the 4th transition - assign action 5 to the ugv and move
#     assert state3_p.get_actions()[1].rtype == param.RobotType.Ground
#     assert state3_p.get_actions()[1].target == 5
#     state_prob_cost = state3_p.transition(state3_p.get_actions()[1])
#     assert len(state_prob_cost) == 2
#     state4_p = list(state_prob_cost.keys())[0]
#     assert state4_p.get_actions()[0].rtype == param.RobotType.Drone
#     assert len(state4_p.get_actions()) == 2
#     assert state4_p.ugvs[0].need_action == False
#     assert state4_p.uavs[0].need_action == True
#     assert state4_p.action_cost == pytest.approx(4.0/param.VEL_RATIO, 0.05)

#     # the 5th transition - assign action 6 to the uav, then move
#     assert state4_p.get_actions()[0].target == 6
#     state_prob_cost = state4_p.transition(state4_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state5_p = list(state_prob_cost.keys())[0]
#     assert state5_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert len(state5_p.get_actions()) == 1
#     assert state5_p.ugvs[0].need_action == True
#     assert state5_p.uavs[0].need_action == False
#     assert state5_p.action_cost == 0.0
    
#     # the 6th transition - assign action 2 to the ugv then move, the drone reach goal first
#     assert state5_p.get_actions()[0].target == 2
#     state_prob_cost = state5_p.transition(state5_p.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state6_p = list(state_prob_cost.keys())[0]
#     assert state6_p.get_actions()[0].rtype == param.RobotType.Drone
#     assert len(state6_p.get_actions()) == 1
#     assert state6_p.ugvs[0].need_action == True
#     assert state6_p.uavs[0].need_action == True
#     assert state6_p.action_cost == 2.0/param.VEL_RATIO

# def test_jsap_transition_dg_anystart():
#     print()
#     starts, goals, graph = graphs.disjoint_unc()
    
#     uavs = [Robot(position=[0.0, 1.0], cur_node=starts[0].id, at_node=False, robot_type=param.RobotType.Drone) \
#                     for _ in range(1)]
#     at_node_ugv = False
#     ugvs = [Robot(position=[5.0, 3.0], cur_node=4, at_node=at_node_ugv,edge=[4,6]) for _ in range(1)]

#     state = jsap.JSAPState(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=uavs,
#                            n_maps=80, useAVP=False, max_uanum=1)
#     if not at_node_ugv: 
#         assert ugvs[0].is_robot_pose_correct(state.vertices_map[6].coord, state.vertices_map[4].coord)
    
    
#     # the first transition - assign action 5 to the drone
#     assert len(state.get_actions()) == 4
#     assert state.get_actions()[0].rtype == param.RobotType.Drone
#     assert state.get_actions()[0].target == 5
#     state_prob_cost = state.transition(state.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state1 = list(state_prob_cost.keys())[0]
#     assert state1.action_cost == 0.0
#     assert state1.heuristic == pytest.approx(4.24, abs=0.05)
#     assert len(state1.get_actions()) == len(state1.state_actions) == len(state1.ugvs_actions[0]) == 2

#     # the second transition - assign action 8 to the ugv, then move
#     assert state1.get_actions()[1].rtype == param.RobotType.Ground
#     assert state1.get_actions()[1].target == 6
#     state_prob_cost = state1.transition(state1.get_actions()[1])
#     assert len(state_prob_cost) == 2
#     state2_p = list(state_prob_cost.keys())[0]    
#     assert len(state2_p.get_actions()) == 3
#     assert state2_p.ugvs[0].need_action == True
#     assert state2_p.uavs[0].need_action == True
#     assert state2_p.action_cost == pytest.approx(2.236/param.VEL_RATIO,abs=0.05)

#     # the third transition - assign action 7 to the uav
#     assert state2_p.get_actions()[1].target == 7
#     assert state2_p.get_actions()[1].rtype == param.RobotType.Drone
#     state_prob_cost = state2_p.transition(state2_p.get_actions()[1])
#     assert len(state_prob_cost) == 1
#     state3_p = list(state_prob_cost.keys())[0]    
#     assert len(state3_p.get_actions()) == 2
#     assert state3_p.ugvs[0].need_action == True
#     assert state3_p.uavs[0].need_action == False
#     assert state3_p.action_cost == 0.0

#     # the 4th transition - assign action 6 to the ugv and move
#     assert state3_p.get_actions()[1].rtype == param.RobotType.Ground
#     assert state3_p.get_actions()[1].target == 6
#     state_prob_cost = state3_p.transition(state3_p.get_actions()[1])
#     assert len(state_prob_cost) == 2
#     state4_p = list(state_prob_cost.keys())[0]
#     assert state4_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert len(state4_p.get_actions()) == 1
#     assert state4_p.ugvs[0].need_action == True
#     assert state4_p.uavs[0].need_action == False

#     # the 5th transition - assign action 3 to the ugv, then move
#     assert state4_p.get_actions()[0].target == 3
#     state_prob_cost = state4_p.transition(state4_p.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state5_p = list(state_prob_cost.keys())[0]
#     assert state5_p.get_actions()[0].rtype == param.RobotType.Drone
#     assert len(state5_p.get_actions()) == 1
#     assert state5_p.ugvs[0].need_action == True
#     assert state5_p.uavs[0].need_action == True
    
#     # the 6th transition - assign action 8 to the uav then move
#     assert state5_p.get_actions()[0].target == 8
#     state_prob_cost = state5_p.transition(state5_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state6_p = list(state_prob_cost.keys())[0]
#     assert state6_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert len(state6_p.get_actions()) == 2
#     assert state6_p.ugvs[0].need_action == True
#     assert state6_p.uavs[0].need_action == False
#     assert state6_p.action_cost == 0.0

#     # the 6th transition - assign action 3 to the ugv then move, the ground reaches goal first
#     assert state6_p.get_actions()[1].target == 3
#     state_prob_cost = state6_p.transition(state6_p.get_actions()[1])
#     assert len(state_prob_cost) == 2
#     state7_p = list(state_prob_cost.keys())[0]
#     assert state7_p.get_actions()[0].rtype == param.RobotType.Drone
#     assert len(state7_p.get_actions()) == 1
#     assert state7_p.ugvs[0].need_action == True
#     assert state7_p.uavs[0].need_action == True


# def test_jsap_transition_2ugvs_sgraph():
#     print()
#     num_uav = 0
#     num_ugv = 2
#     max_uanum = 1
#     starts, goals, graph = graphs.s_graph_2goals()
#     uavs = []
#     ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(num_ugv)]
#     useAVP = False
#     state = jsap.JSAPState(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=uavs, n_maps=60,\
#                            useAVP=useAVP, max_uanum=max_uanum)
#     assert state.heuristic == pytest.approx(17.66, 0.05)
#     assert state.history.get_data_length() == len(graph.vertices)
#     assert len(state.state_actions) == len(state.get_actions()) == 2
#     assert all(uav.need_action == True for uav in state.uavs)
#     assert all(ugv.need_action == True for ugv in state.ugvs)
#     assert len(state.assigned_pois) == 0
# #####++++++ Error in copy function: The action target is 6 from 10 of UGV 1 in the copy function
# #####++++++ And robot 0's last node: 2 on edge [2, 6]
# #####++++++ And robot 1's last node: 10 on edge []
# # The current actions of UGV 0 is: [6, 9, 10]
# # The current actions of UGV 1 is: [6, 9, 10]
# # The poses of UGV 0 is: [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [4.0, 4.0], [2.586, 2.586]] and current pose [2.586 2.586]
# # The poses of UGV 1 is: [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [4.0, 4.0], [6.0, 4.0]] and current pose [6. 4.]


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

