import pytest
from sctp import sctp_graphs as graphs
# from sctp import core
from sctp.robot import Robot

from sctp import param, jsap
# from sctp.utils import plotting, paths
# import matplotlib.pyplot as plt

def test_jsap_transition_lgraph():
    print()
    starts, goals, l_graph = graphs.linear_graph_unc()
    drones = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True, robot_type=param.RobotType.Drone) for i in range(1)]
    ugvs = [Robot(position=[0.0, 0.0], at_node=True, cur_node=starts[0].id) for _ in range(1)]
    
    state = jsap.JSAPState(graph=l_graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=drones,\
                n_maps=80, useAVP=False, max_uanum=1)
    assert state.heuristic == 15.0
    assert state.history.get_data_length() == len(l_graph.vertices)
    assert len(state.state_actions) == len(state.get_actions()) == 2
    assert all(uav.need_action == True for uav in state.uavs)
    assert all(ugv.need_action == True for ugv in state.ugvs)
    assert len(state.assigned_pois) == 0

    state2 = state.copy()
    assert state2 != state
    for i in range(len(state2.uavs)):
        assert state2.uavs[i] != state.uavs[i]
    assert state2.state_actions == []
    assert state2.uav_actions == state.uav_actions
    assert state2.ugvs_actions[0] == state.ugvs_actions[0]
    assert state2.assigned_pois == state.assigned_pois
    
    # the first transition - assign action to the drone
    assert len(state.get_actions()) == 2
    assert state.get_actions()[0].rtype == param.RobotType.Drone
    assert state.get_actions()[0].target == 4
    state_prob_cost = state.transition(state.get_actions()[0])
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    assert state1.action_cost == 0.0
    assert state1.heuristic == 15.0
    assert len(state1.get_actions()) == len(state1.state_actions) == len(state1.ugvs_actions[0]) == 1

    # the second transition - assign action to the ugv then move
    assert state1.get_actions()[0].target == 4
    state_prob_cost = state1.transition(state1.get_actions()[0])
    assert state1.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state_prob_cost) == 2
    state2_p = list(state_prob_cost.keys())[0]
    assert state2_p.get_actions()[0].rtype == param.RobotType.Drone
    assert len(state2_p.ugvs_actions[0]) == 2
    assert len(state2_p.get_actions()) == 1
    assert state2_p.ugvs[0].need_action == True
    assert state2_p.uavs[0].need_action == True
    assert state2_p.action_cost == 2.5/param.VEL_RATIO
    assert state2_p.heuristic == pytest.approx(15.0-state2_p.action_cost, 0.05)
    assert state2_p.noway2goal == False
    
    state2_b = list(state_prob_cost.keys())[1]
    assert state2_b.ugvs[0].need_action == True
    assert state2_b.uavs[0].need_action == True
    assert state2_b.action_cost == 2.5/param.VEL_RATIO
    assert state2_b.heuristic == param.NOWAY_PEN
    assert state2_b.noway2goal == True
    assert state2_b.is_goal_state == True
    
    # the third transition - assign action to the uav 
    state_prob_cost = state2_p.transition(state2_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state3_p = list(state_prob_cost.keys())[0]
    assert state3_p.get_actions()[0].rtype == param.RobotType.Ground
    assert state3_p.ugvs[0].need_action == True
    assert state3_p.uavs[0].need_action == False
    assert state3_p.action_cost == 0.0
    
    # the fourth transition - assign action 1 to the ugv then move 
    state_prob_cost = state3_p.transition(state3_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state4_p = list(state_prob_cost.keys())[0]
    assert state4_p.get_actions()[0].rtype == param.RobotType.Ground
    assert state4_p.ugvs[0].need_action == True
    assert state4_p.uavs[0].need_action == False
    assert state4_p.action_cost == (2.5/param.VEL_RATIO)
    
    # action 4
    state_prob_cost = state3_p.transition(state3_p.get_actions()[1])
    assert len(state_prob_cost) == 1
    state4_p = list(state_prob_cost.keys())[0]
    assert state4_p.get_actions()[0].rtype == param.RobotType.Ground
    assert state4_p.ugvs[0].need_action == True
    assert state4_p.uavs[0].need_action == False
    assert state4_p.action_cost == (2.5-2.5/param.VEL_RATIO)
    
    # the fifth transition - assign action 2 to the ugv, then move
    # then uav 0 reach node v5 first/ reset action both ugvs and uavs 
    assert state4_p.get_actions()[0].target == 2
    state_prob_cost = state4_p.transition(state4_p.get_actions()[0])
    assert len(state_prob_cost) == 2

    # blocked state
    state5_b = list(state_prob_cost.keys())[1]
    assert state5_b.get_actions()[0].rtype == param.RobotType.Drone
    assert state5_b.ugvs[0].need_action == True
    assert state5_b.uavs[0].need_action == True
    assert state5_b.action_cost == pytest.approx(2.5/param.VEL_RATIO, 0.05)
    assert state5_b.uavs[0].last_node == 5
    assert state5_b.noway2goal == True 
    assert state5_b.is_goal_state == True
    

    state5_p = list(state_prob_cost.keys())[0]
    assert state5_p.get_actions()[0].rtype == param.RobotType.Drone
    assert state5_p.ugvs[0].need_action == True
    assert state5_p.uavs[0].need_action == True
    assert state5_p.action_cost == pytest.approx(2.5/param.VEL_RATIO, 0.05)
    assert state5_p.uavs[0].last_node == 5
    
    # the sixth transition - assign action 3 to the uav
    assert state5_p.get_actions()[0].target == 3
    state_prob_cost = state5_p.transition(state5_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state6_p = list(state_prob_cost.keys())[0]
    assert state6_p.get_actions()[0].rtype == param.RobotType.Ground
    assert state6_p.ugvs[0].need_action == True
    assert state6_p.uavs[0].need_action == False
    assert state6_p.action_cost == 0.0
    assert state6_p.uavs[0].last_node == 5
    assert len(state6_p.get_actions()) == 2
    assert state6_p.get_actions()[1].target == 2
    
    # the 7th transition - assign action 2 to the ugv then move
    # both teams ugv and uav reach their targets
    state_prob_cost = state6_p.transition(state6_p.get_actions()[1])
    assert len(state_prob_cost) == 1
    state7_p = list(state_prob_cost.keys())[0]
    assert state7_p.ugvs[0].last_node == 2
    assert state7_p.ugvs[0].need_action == False
    assert state7_p.uavs[0].need_action == True
    assert state7_p.action_cost == pytest.approx(5.0/param.VEL_RATIO, 0.05)
    assert state7_p.get_actions()[0].rtype == param.RobotType.Drone
    
    # the 8th transition - assign wait action for the drone
    state_prob_cost = state7_p.transition(state7_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state8_p = list(state_prob_cost.keys())[0]
    assert len(state8_p.get_actions()) == 1
    assert state8_p.ugvs[0].last_node == 2
    assert state8_p.ugvs[0].need_action == True
    assert state8_p.uavs[0].need_action == False
    assert state8_p.action_cost == pytest.approx(0, param.APPROX_TIME)
    assert state8_p.get_actions()[0].rtype == param.RobotType.Ground
    
    # the 9th transition - assign action 5 to the ugv then move
    state_prob_cost = state8_p.transition(state8_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state9_p = list(state_prob_cost.keys())[0]
    # print(f"The position of ugv: {state9_p.ugvs[0].cur_pose}, last node: {state9_p.ugvs[0].last_node}")
    assert len(state9_p.get_actions()) == 1
    assert state9_p.ugvs[0].last_node == 5
    assert state9_p.uavs[0].last_node == 3
    assert state9_p.ugvs[0].need_action == True
    assert state9_p.uavs[0].need_action == False
    assert state9_p.action_cost == pytest.approx(5.0, param.APPROX_TIME)
    assert state9_p.get_actions()[0].rtype == param.RobotType.Ground
    
    # the 10th transition - assign action 3 to the ugv then move
    state_prob_cost = state9_p.transition(state9_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state10_p = list(state_prob_cost.keys())[0]
    print(f"The position of ugv: {state10_p.ugvs[0].cur_pose}, last node: {state10_p.ugvs[0].last_node}")
    assert len(state10_p.get_actions()) == 1
    assert state10_p.ugvs[0].last_node == 3
    assert state10_p.uavs[0].last_node == 3
    assert state10_p.ugvs[0].need_action == True
    assert state10_p.uavs[0].need_action == False
    assert state10_p.action_cost == pytest.approx(5.0, param.APPROX_TIME)
    assert state10_p.is_goal_state == True
    assert state10_p.noway2goal == False

def test_jsap_transition_1uav1ugv_dg():
    print()
    starts, goals, graph = graphs.disjoint_unc()
    uavs = [Robot(position=[0.0, 0.0], cur_node=starts[0].id, at_node=True, robot_type=param.RobotType.Drone) \
                    for _ in range(1)]
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[0].id, at_node=True) for _ in range(1)]

    state = jsap.JSAPState(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=uavs,
                           n_maps=80, useAVP=False, max_uanum=1)
    
    assert state.heuristic == 8.0
    assert state.history.get_data_length() == len(graph.vertices)
    assert len(state.state_actions) == len(state.get_actions()) == 4
    assert all(uav.need_action == True for uav in state.uavs)
    assert all(ugv.need_action == True for ugv in state.ugvs)
    assert len(state.assigned_pois) == 0

    # the first transition - assign action 5 to the drone
    assert len(state.get_actions()) == 4
    assert state.get_actions()[0].rtype == param.RobotType.Drone
    assert state.get_actions()[0].target == 5
    state_prob_cost = state.transition(state.get_actions()[0])
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    assert state1.action_cost == 0.0
    assert state1.heuristic == 8.0
    assert len(state1.get_actions()) == len(state1.state_actions) == len(state1.ugvs_actions[0]) == 2

    # the second transition - assign action 8 to the ugv, then move
    assert state1.get_actions()[1].rtype == param.RobotType.Ground
    assert state1.get_actions()[1].target == 8
    state_prob_cost = state1.transition(state1.get_actions()[1])
    assert len(state_prob_cost) == 2
    state2_p = list(state_prob_cost.keys())[0]    
    assert len(state2_p.get_actions()) == 3
    assert state2_p.ugvs[0].need_action == True
    assert state2_p.uavs[0].need_action == True
    assert state2_p.action_cost == 2.0/param.VEL_RATIO

    # the third transition - assign action 6 to the uav
    assert state2_p.get_actions()[0].target == 6
    assert state2_p.get_actions()[0].rtype == param.RobotType.Drone
    state_prob_cost = state2_p.transition(state2_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state3_p = list(state_prob_cost.keys())[0]    
    assert len(state3_p.get_actions()) == 2
    assert state3_p.ugvs[0].need_action == True
    assert state3_p.uavs[0].need_action == False
    assert state3_p.action_cost == 0.0

    # the 4th transition - assign action 8 to the ugv and move
    assert state3_p.get_actions()[1].rtype == param.RobotType.Ground
    assert state3_p.get_actions()[1].target == 8
    state_prob_cost = state3_p.transition(state3_p.get_actions()[1])
    assert len(state_prob_cost) == 2
    state4_p = list(state_prob_cost.keys())[0]
    assert state4_p.get_actions()[0].rtype == param.RobotType.Drone
    assert len(state4_p.get_actions()) == 2
    assert state4_p.ugvs[0].need_action == True
    assert state4_p.uavs[0].need_action == True
    assert state4_p.action_cost == pytest.approx(1.491, 0.05)

    # the 5th transition - assign action 7 to the uav
    assert state4_p.get_actions()[0].target == 7
    state_prob_cost = state4_p.transition(state4_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state5_p = list(state_prob_cost.keys())[0]
    assert state5_p.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state5_p.get_actions()) == 2
    assert state5_p.ugvs[0].need_action == True
    assert state5_p.uavs[0].need_action == False
    assert state5_p.action_cost == 0.0
    
    # the 6th transition - assign action 8 to the ugv then move, the drone reach goal first
    assert state5_p.get_actions()[1].target == 8
    state_prob_cost = state5_p.transition(state5_p.get_actions()[1])
    assert len(state_prob_cost) == 2
    state6_p = list(state_prob_cost.keys())[0]
    # print(f"The position of ugv: {state6_p.ugvs[0].cur_pose}, last node: {state6_p.ugvs[0].last_node}")
    # print(f"The position of drone: {state6_p.uavs[0].cur_pose}, last node: {state6_p.uavs[0].last_node}")
    assert state6_p.get_actions()[0].rtype == param.RobotType.Drone
    assert len(state6_p.get_actions()) == 1
    assert state6_p.ugvs[0].need_action == True
    assert state6_p.uavs[0].need_action == True
    assert state6_p.action_cost == 2.0/param.VEL_RATIO

    # the 7th transition - assign action 8 to the uav 
    assert state6_p.get_actions()[0].target == 8
    state_prob_cost = state6_p.transition(state6_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state7_p = list(state_prob_cost.keys())[0]
    assert state7_p.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state7_p.get_actions()) == 2
    assert state7_p.ugvs[0].need_action == True
    assert state7_p.uavs[0].need_action == False

    # the 8th transition - assign action to the ugv, then move 
    assert state7_p.get_actions()[1].target == 8
    state_prob_cost = state7_p.transition(state7_p.get_actions()[1])
    assert len(state_prob_cost) == 2
    state8_p = list(state_prob_cost.keys())[0]
    assert state8_p.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state8_p.get_actions()) == 1
    assert state8_p.ugvs[0].need_action == True
    assert state8_p.uavs[0].need_action == True
    assert state8_p.action_cost == pytest.approx(0.005, abs=0.001)

    # the 9th transition - assign action 4 to the ugv 
    assert state8_p.get_actions()[0].target == 4
    state_prob_cost = state8_p.transition(state8_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state9_p = list(state_prob_cost.keys())[0]
    assert state9_p.get_actions()[0].rtype == param.RobotType.Drone
    assert len(state9_p.get_actions()) == 1
    assert state9_p.ugvs[0].need_action == False
    assert state9_p.uavs[0].need_action == True
    assert state9_p.action_cost == 0.0

    # the 10th transition - assign action 3 (goal) to the uav, then move
    assert state9_p.get_actions()[0].target == 3
    state_prob_cost = state9_p.transition(state9_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state10_p = list(state_prob_cost.keys())[0]
    assert state10_p.get_actions()[0].rtype == param.RobotType.Drone
    assert len(state10_p.get_actions()) == 1
    assert state10_p.ugvs[0].need_action == False
    assert state10_p.uavs[0].need_action == True
    
    # the 11th transition - assign action to the uav, (staying at goal) then move ugv
    assert state10_p.get_actions()[0].target == 3
    state_prob_cost = state10_p.transition(state10_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state11_p = list(state_prob_cost.keys())[0]
    assert state11_p.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state11_p.get_actions()) == 1
    assert state11_p.ugvs[0].need_action == True
    assert state11_p.uavs[0].need_action == False

    # the 12th transition - assign action to the ugv, then move
    assert state11_p.get_actions()[0].target == 6
    state_prob_cost = state11_p.transition(state11_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state12_p = list(state_prob_cost.keys())[0]
    assert state12_p.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state12_p.get_actions()) == 1
    assert state12_p.ugvs[0].need_action == True
    assert state12_p.uavs[0].need_action == False
    assert state12_p.action_cost == pytest.approx(2.828, 0.05)

    # the 13th transition - assign action to the ugv, then move
    assert state12_p.get_actions()[0].target == 3
    state_prob_cost = state12_p.transition(state12_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state13_p = list(state_prob_cost.keys())[0]
    assert state13_p.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state13_p.get_actions()) == 1
    assert state13_p.ugvs[0].need_action == True
    assert state13_p.uavs[0].need_action == False
    assert state13_p.action_cost == pytest.approx(2.828, 0.05)
    assert state13_p.is_goal_state == True
    assert state13_p.noway2goal == False
    
def test_jsap_transition_dg_finish_sametime():
    print()
    starts, goals, graph = graphs.disjoint_unc()
    uavs = [Robot(position=[0.0, 0.0], cur_node=starts[0].id, at_node=True, robot_type=param.RobotType.Drone) \
                    for _ in range(1)]
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[0].id, at_node=True) for _ in range(1)]

    state = jsap.JSAPState(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=uavs,
                           n_maps=80, useAVP=False, max_uanum=1)

    # the first transition - assign action 5 to the drone
    assert len(state.get_actions()) == 4
    assert state.get_actions()[0].rtype == param.RobotType.Drone
    assert state.get_actions()[0].target == 5
    state_prob_cost = state.transition(state.get_actions()[0])
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    assert state1.action_cost == 0.0
    assert state1.heuristic == 8.0
    assert len(state1.get_actions()) == len(state1.state_actions) == len(state1.ugvs_actions[0]) == 2

    # the second transition - assign action 8 to the ugv, then move
    assert state1.get_actions()[0].rtype == param.RobotType.Ground
    assert state1.get_actions()[0].target == 5
    state_prob_cost = state1.transition(state1.get_actions()[0])
    assert len(state_prob_cost) == 2
    state2_p = list(state_prob_cost.keys())[0]    
    assert len(state2_p.get_actions()) == 3
    assert state2_p.ugvs[0].need_action == True
    assert state2_p.uavs[0].need_action == True
    assert state2_p.action_cost == 2.0/param.VEL_RATIO

    # the third transition - assign action 6 to the uav
    assert state2_p.get_actions()[1].target == 7
    assert state2_p.get_actions()[1].rtype == param.RobotType.Drone
    state_prob_cost = state2_p.transition(state2_p.get_actions()[1])
    assert len(state_prob_cost) == 1
    state3_p = list(state_prob_cost.keys())[0]    
    assert len(state3_p.get_actions()) == 2
    assert state3_p.ugvs[0].need_action == True
    assert state3_p.uavs[0].need_action == False
    assert state3_p.action_cost == 0.0

    # the 4th transition - assign action 5 to the ugv and move
    assert state3_p.get_actions()[1].rtype == param.RobotType.Ground
    assert state3_p.get_actions()[1].target == 5
    state_prob_cost = state3_p.transition(state3_p.get_actions()[1])
    assert len(state_prob_cost) == 2
    state4_p = list(state_prob_cost.keys())[0]
    assert state4_p.get_actions()[0].rtype == param.RobotType.Drone
    assert len(state4_p.get_actions()) == 2
    assert state4_p.ugvs[0].need_action == False
    assert state4_p.uavs[0].need_action == True
    assert state4_p.action_cost == pytest.approx(4.0/param.VEL_RATIO, 0.05)

    # the 5th transition - assign action 6 to the uav, then move
    assert state4_p.get_actions()[0].target == 6
    state_prob_cost = state4_p.transition(state4_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state5_p = list(state_prob_cost.keys())[0]
    assert state5_p.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state5_p.get_actions()) == 1
    assert state5_p.ugvs[0].need_action == True
    assert state5_p.uavs[0].need_action == False
    assert state5_p.action_cost == 0.0
    
    # the 6th transition - assign action 2 to the ugv then move, the drone reach goal first
    assert state5_p.get_actions()[0].target == 2
    state_prob_cost = state5_p.transition(state5_p.get_actions()[0])
    assert len(state_prob_cost) == 2
    state6_p = list(state_prob_cost.keys())[0]
    assert state6_p.get_actions()[0].rtype == param.RobotType.Drone
    assert len(state6_p.get_actions()) == 1
    assert state6_p.ugvs[0].need_action == True
    assert state6_p.uavs[0].need_action == True
    assert state6_p.action_cost == 2.0/param.VEL_RATIO

def test_jsap_transition_dg_anystart():
    print()
    starts, goals, graph = graphs.disjoint_unc()
    
    uavs = [Robot(position=[0.0, 1.0], cur_node=starts[0].id, at_node=False, robot_type=param.RobotType.Drone) \
                    for _ in range(1)]
    at_node_ugv = False
    ugvs = [Robot(position=[5.0, 3.0], cur_node=4, at_node=at_node_ugv,edge=[4,6]) for _ in range(1)]

    state = jsap.JSAPState(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=uavs,
                           n_maps=80, useAVP=False, max_uanum=1)
    if not at_node_ugv: 
        assert ugvs[0].is_robot_pose_correct(state.vertices_map[6].coord, state.vertices_map[4].coord)
    
    
    # the first transition - assign action 5 to the drone
    assert len(state.get_actions()) == 4
    assert state.get_actions()[0].rtype == param.RobotType.Drone
    assert state.get_actions()[0].target == 5
    state_prob_cost = state.transition(state.get_actions()[0])
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    assert state1.action_cost == 0.0
    assert state1.heuristic == pytest.approx(4.24, abs=0.05)
    assert len(state1.get_actions()) == len(state1.state_actions) == len(state1.ugvs_actions[0]) == 2

    # the second transition - assign action 8 to the ugv, then move
    assert state1.get_actions()[1].rtype == param.RobotType.Ground
    assert state1.get_actions()[1].target == 6
    state_prob_cost = state1.transition(state1.get_actions()[1])
    assert len(state_prob_cost) == 2
    state2_p = list(state_prob_cost.keys())[0]    
    assert len(state2_p.get_actions()) == 3
    assert state2_p.ugvs[0].need_action == True
    assert state2_p.uavs[0].need_action == True
    assert state2_p.action_cost == pytest.approx(2.236/param.VEL_RATIO,abs=0.05)

    # the third transition - assign action 7 to the uav
    assert state2_p.get_actions()[1].target == 7
    assert state2_p.get_actions()[1].rtype == param.RobotType.Drone
    state_prob_cost = state2_p.transition(state2_p.get_actions()[1])
    assert len(state_prob_cost) == 1
    state3_p = list(state_prob_cost.keys())[0]    
    assert len(state3_p.get_actions()) == 2
    assert state3_p.ugvs[0].need_action == True
    assert state3_p.uavs[0].need_action == False
    assert state3_p.action_cost == 0.0

    # the 4th transition - assign action 6 to the ugv and move
    assert state3_p.get_actions()[1].rtype == param.RobotType.Ground
    assert state3_p.get_actions()[1].target == 6
    state_prob_cost = state3_p.transition(state3_p.get_actions()[1])
    assert len(state_prob_cost) == 2
    state4_p = list(state_prob_cost.keys())[0]
    assert state4_p.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state4_p.get_actions()) == 1
    assert state4_p.ugvs[0].need_action == True
    assert state4_p.uavs[0].need_action == False

    # the 5th transition - assign action 3 to the ugv, then move
    assert state4_p.get_actions()[0].target == 3
    state_prob_cost = state4_p.transition(state4_p.get_actions()[0])
    assert len(state_prob_cost) == 2
    state5_p = list(state_prob_cost.keys())[0]
    assert state5_p.get_actions()[0].rtype == param.RobotType.Drone
    assert len(state5_p.get_actions()) == 1
    assert state5_p.ugvs[0].need_action == True
    assert state5_p.uavs[0].need_action == True
    
    # the 6th transition - assign action 8 to the uav then move
    assert state5_p.get_actions()[0].target == 8
    state_prob_cost = state5_p.transition(state5_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state6_p = list(state_prob_cost.keys())[0]
    assert state6_p.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state6_p.get_actions()) == 2
    assert state6_p.ugvs[0].need_action == True
    assert state6_p.uavs[0].need_action == False
    assert state6_p.action_cost == 0.0

    # the 6th transition - assign action 3 to the ugv then move, the ground reaches goal first
    assert state6_p.get_actions()[1].target == 3
    state_prob_cost = state6_p.transition(state6_p.get_actions()[1])
    assert len(state_prob_cost) == 2
    state7_p = list(state_prob_cost.keys())[0]
    assert state7_p.get_actions()[0].rtype == param.RobotType.Drone
    assert len(state7_p.get_actions()) == 1
    assert state7_p.ugvs[0].need_action == True
    assert state7_p.uavs[0].need_action == True


def test_jsap_transition_2ugvs_sgraph():
    print()
    num_uav = 0
    num_ugv = 2
    max_uanum = 1
    starts, goals, graph = graphs.s_graph_2goals()
    uavs = []
    ugvs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, at_node=True) for i in range(num_ugv)]
    useAVP = False
    state = jsap.JSAPState(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=uavs, n_maps=60,\
                           useAVP=useAVP, max_uanum=max_uanum)
    assert state.heuristic == pytest.approx(17.66, 0.05)
    assert state.history.get_data_length() == len(graph.vertices)
    assert len(state.state_actions) == len(state.get_actions()) == 2
    assert all(uav.need_action == True for uav in state.uavs)
    assert all(ugv.need_action == True for ugv in state.ugvs)
    assert len(state.assigned_pois) == 0
#####++++++ Error in copy function: The action target is 6 from 10 of UGV 1 in the copy function
#####++++++ And robot 0's last node: 2 on edge [2, 6]
#####++++++ And robot 1's last node: 10 on edge []
# The current actions of UGV 0 is: [6, 9, 10]
# The current actions of UGV 1 is: [6, 9, 10]
# The poses of UGV 0 is: [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [4.0, 4.0], [2.586, 2.586]] and current pose [2.586 2.586]
# The poses of UGV 1 is: [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [4.0, 4.0], [6.0, 4.0]] and current pose [6. 4.]


    # the first transition - assign action to the first Ground
    assert state.get_actions()[0].rtype == param.RobotType.Ground
    assert state.get_actions()[1].target == 7
    state_prob_cost = state.transition(state.get_actions()[1])
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    assert len(state1.get_actions()) == 2
    assert state1.get_actions()[0].rtype == param.RobotType.Ground
    assert state1.action_cost == 0.0
    assert state1.heuristic == pytest.approx(17.66, 0.05)
    assert state1.ugvs[0].need_action == False
    assert state1.ugvs[1].need_action == True
        
    # the second transition - assign action 7 to the second ugv
    assert state1.get_actions()[1].target == 7
    state_prob_cost = state1.transition(state1.get_actions()[1])
    assert len(state_prob_cost) == 2
    state2_p = list(state_prob_cost.keys())[0]
    assert len(state2_p.get_actions()) == 1
    assert state2_p.get_actions()[0].rtype == param.RobotType.Ground
    assert state2_p.action_cost == 2.0
    assert state2_p.heuristic == 16.0
    assert state2_p.ugvs[0].need_action == True
    assert state2_p.ugvs[1].need_action == False

    # the third transition - assign action 3 to ugv 0, then move
    assert state2_p.get_actions()[0].target == 3
    assert state2_p.get_actions()[0].robotID == 0
    state_prob_cost = state2_p.transition(state2_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state3_p = list(state_prob_cost.keys())[0]
    assert len(state3_p.get_actions()) == 1
    assert state3_p.get_actions()[0].rtype == param.RobotType.Ground
    assert state3_p.action_cost == 0.0
    assert state3_p.ugvs[0].need_action == False
    assert state3_p.ugvs[1].need_action == True

    # the 4th transition - assign action 3 to ugv 1, then move
    assert state3_p.get_actions()[0].target == 3
    assert state3_p.get_actions()[0].robotID == 1
    state_prob_cost = state3_p.transition(state3_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state4_p = list(state_prob_cost.keys())[0]
    assert state4_p.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state4_p.get_actions()) == len(state4_p.state_actions) == 2
    assert state4_p.get_actions()[0].robotID == state4_p.get_actions()[0].robotID == 0
    assert state4_p.action_cost == 2.0
    assert state4_p.ugvs[0].need_action == True
    assert state4_p.ugvs[1].need_action == False   

    # the 5th transition - assign action 8 to ugv 0
    assert state4_p.get_actions()[0].target == 8
    assert state4_p.get_actions()[1].target == 11
    assert state4_p.ugvs_actions[0][0].robotID == state4_p.ugvs_actions[0][1].robotID == 0
    assert state4_p.ugvs_actions[1][0].robotID == 1
    assert state4_p.ugvs_actions[1][0].target == 3
    state4_p.get_actions()[0].robotID == state4_p.get_actions()[1].robotID == 0
    state_prob_cost = state4_p.transition(state4_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state5_p = list(state_prob_cost.keys())[0]
    assert state5_p.get_actions()[0].rtype == param.RobotType.Ground
    assert len(state5_p.get_actions()) == len(state5_p.state_actions) == 2
    assert state5_p.action_cost == 0.0
    assert state5_p.ugvs[0].need_action == False
    assert state5_p.ugvs[1].need_action == True


    # the 6th transition - assign action 8 to the uav 1
    assert state5_p.get_actions()[0].target == 8
    assert state5_p.get_actions()[1].target == 11
    state_prob_cost = state5_p.transition(state5_p.get_actions()[0])
    assert len(state_prob_cost) == 2
    state6_p = list(state_prob_cost.keys())[0]
    assert len(state6_p.get_actions()) == len(state6_p.state_actions) == 1
    assert state6_p.get_actions()[0].rtype == param.RobotType.Ground
    assert state6_p.action_cost == 2.0
    assert state6_p.ugvs[0].need_action == True
    assert state6_p.ugvs[1].need_action == False    
    
    # the 7th transition - assign action 2 to ugv 0
    assert state6_p.get_actions()[0].target == 2
    state_prob_cost = state6_p.transition(state6_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state7_p = list(state_prob_cost.keys())[0]
    assert len(state7_p.get_actions()) == len(state7_p.state_actions) == 1
    assert state7_p.get_actions()[0].rtype == param.RobotType.Ground
    assert state7_p.action_cost == 0.0
    assert state7_p.ugvs[0].need_action == False
    assert state7_p.ugvs[1].need_action == True    
# The poses of UGV 0 is: [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [4.0, 4.0], [2.586, 2.586]] and current pose [2.586 2.586]
# The poses of UGV 1 is: [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [4.0, 4.0], [6.0, 4.0]] and current pose [6. 4.]
 

    # the 8th transition - assign action 2 to UGV 1
    assert state7_p.get_actions()[0].target == 2
    state_prob_cost = state7_p.transition(state7_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state8_p = list(state_prob_cost.keys())[0]
    assert len(state8_p.get_actions()) == len(state8_p.state_actions) == 3
    assert state8_p.get_actions()[0].rtype == param.RobotType.Ground
    assert state8_p.action_cost == 2.0
    assert state8_p.ugvs[0].need_action == True
    assert state8_p.ugvs[1].need_action == False    

    # the 9th transition - assign action 6 to UGV 0, then move
    assert state8_p.get_actions()[0].target == 6
    state_prob_cost = state8_p.transition(state8_p.get_actions()[0])
    assert len(state_prob_cost) == 1
    state9_p = list(state_prob_cost.keys())[0]
    assert len(state9_p.get_actions()) == len(state9_p.state_actions) == 3
    assert state9_p.get_actions()[0].rtype == param.RobotType.Ground
    assert state9_p.action_cost == 0.0
    assert state9_p.ugvs[0].need_action == False
    assert state9_p.ugvs[1].need_action == True    

    # print(f"The pose of UGV 0 is: {state9_p.ugvs[0].cur_pose}, last node: {state9_p.ugvs[0].last_node}")
    # print(f"The pose of UGV 1 is: {state9_p.ugvs[1].cur_pose}, last node: {state9_p.ugvs[1].last_node}")


    # the 10th transition - assign action 9 to UGV 1, then move
    assert state9_p.get_actions()[2].target == 10
    # for action in state9_p.get_actions():
    #     print(f"Action target: {action.target} of UGV {action.robotID} from {action.start_pose}")
    state_prob_cost = state9_p.transition(state9_p.get_actions()[2])
    # assert len(state_prob_cost) == 2
    # state10_p = list(state_prob_cost.keys())[0]
    # assert len(state10_p.get_actions()) == len(state10_p.state_actions) == 1
    # assert state10_p.get_actions()[0].rtype == param.RobotType.Ground
    # assert state10_p.action_cost == 2.0
    # assert state10_p.ugvs[0].need_action == False
    # assert state10_p.ugvs[1].need_action == True
    

#     # the 11th transition - assign action to the second uav, then move
#     assert state10_p.get_actions()[0].target == 3
#     state_prob_cost = state10_p.transition(state10_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state11_p = list(state_prob_cost.keys())[0]
#     assert state11_p.get_actions()[0].rtype == param.RobotType.Drone
#     # assert state11_p.action_cost == 0.0
#     assert state11_p.ugvs[0].need_action == False
#     assert state11_p.uavs[0].need_action == False
#     assert state11_p.uavs[1].need_action == True    
#     assert len(state11_p.get_actions()) == len(state11_p.state_actions) == 1

#     # the 12th transition - assign action to the second uav, then move
#     assert state11_p.get_actions()[0].target == 3
#     state_prob_cost = state11_p.transition(state11_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state12_p = list(state_prob_cost.keys())[0]
#     assert state12_p.get_actions()[0].rtype == param.RobotType.Drone
#     assert state12_p.action_cost > 0.0
#     assert state12_p.ugvs[0].need_action == False
#     assert state12_p.uavs[0].need_action == True
#     assert state12_p.uavs[1].need_action == False    
#     assert len(state12_p.get_actions()) == len(state12_p.state_actions) == 1

#     # the 13th transition - assign wait action to the first uav, then move
#     assert state12_p.get_actions()[0].target == 3
#     state_prob_cost = state12_p.transition(state12_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state13_p = list(state_prob_cost.keys())[0]
#     assert state13_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert state13_p.action_cost > 0.0
#     assert state13_p.ugvs[0].need_action == True
#     assert state13_p.uavs[0].need_action == False
#     assert state13_p.uavs[1].need_action == False    
#     assert len(state13_p.get_actions()) == len(state13_p.state_actions) == 1

#     # the 14th transition - assign action to the ugv, then move
#     assert state13_p.get_actions()[0].target == 6
#     state_prob_cost = state13_p.transition(state13_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state14_p = list(state_prob_cost.keys())[0]
#     assert state14_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert state14_p.action_cost == pytest.approx(2.828, 0.05)
#     assert state14_p.ugvs[0].need_action == True
#     assert state14_p.uavs[0].need_action == False
#     assert state14_p.uavs[1].need_action == False    
#     assert len(state14_p.get_actions()) == len(state14_p.state_actions) == 1

#     # the 14th transition - assign action to the ugv, then move
#     assert state14_p.get_actions()[0].target == 3
#     state_prob_cost = state14_p.transition(state14_p.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state15_p = list(state_prob_cost.keys())[0]
#     assert state15_p.get_actions()[0].rtype == param.RobotType.Ground
#     assert state15_p.action_cost == pytest.approx(2.828, 0.05)
#     assert state15_p.ugvs[0].need_action == True
#     assert state15_p.uavs[0].need_action == False
#     assert state15_p.uavs[1].need_action == False    
#     assert len(state15_p.get_actions()) == len(state15_p.state_actions) == 1
#     assert state15_p.is_goal_state == True

# def test_stateDecPrior_transition_1uav2ugv_reachtarget_sametime():
#     print()
#     num_uav = 1
#     num_ugv = 2
#     starts, goals, graph = graphs.s_graph_2goals()
#     uavs = [Robot(position=[0.0, 0.0], cur_node=1, at_node=True, robot_type=param.RobotType.Drone) for i in range(num_uav)]
#     ugv1 = Robot(position=[0.0, 0.0], cur_node=1, at_node=True)
#     ugv2 = Robot(position=[0.0, 0.0], cur_node=1, at_node=True)
#     # ugv2.last_node = 1
#     ugvs = [ugv1, ugv2]

#     state = dec_prior.StateDecPrior(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=uavs)
#     # state.history.add_history(core.Action(rtype=param.RobotType.Drone, target=6), outcome=param.EventOutcome.TRAV)
#     assert all(uav.need_action == True for uav in state.uavs)
#     assert all(ugv.need_action == True for ugv in state.ugvs)
#     assert len(state.assigned_pois) == 0

#     # the first transition - assign action to the drone
#     assert state.get_actions()[0].rtype == param.RobotType.Drone
#     assert state.get_actions()[6].target == 12
#     state_prob_cost = state.transition(state.get_actions()[6])
#     assert len(state_prob_cost) == 1
#     state1 = list(state_prob_cost.keys())[0]
#     assert state1.action_cost == 0.0
#     assert state1.uavs[0].need_action == False
#     assert state1.ugvs[0].need_action == True
#     assert state1.ugvs[1].need_action == True
#     assert len(state1.get_actions()) == len(state1.state_actions) == 1

#     # the second transition - assign action to the ugv 1
#     assert state1.get_actions()[0].rtype == param.RobotType.Ground
#     assert state1.get_actions()[0].target == 6
#     state_prob_cost = state1.transition(state1.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state2 = list(state_prob_cost.keys())[0]
#     assert state2.action_cost == 0.0
#     # assert state2.heuristic == pytest.approx(8.0+8.243, 0.05)
#     assert state2.uavs[0].need_action == False
#     assert state2.ugvs[0].need_action == False
#     assert state2.ugvs[1].need_action == True
#     assert len(state2.get_actions()) == len(state2.state_actions) == 1
    
#     # the third transition - assign action to the ugv 2, then move
#     assert state2.get_actions()[0].rtype == param.RobotType.Ground
#     assert state2.get_actions()[0].target == 6
#     state_prob_cost = state2.transition(state2.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state3 = list(state_prob_cost.keys())[0]
#     assert state3.uavs[0].need_action == True
#     assert state3.ugvs[0].need_action == True
#     assert state3.ugvs[1].need_action == True
#     assert len(state3.get_actions()) == len(state3.state_actions) == 6
    
#     # the 4th transition - assign action to the uav
#     assert state3.get_actions()[0].rtype == param.RobotType.Drone
#     assert state3.get_actions()[1].target == 7
#     state_prob_cost = state3.transition(state3.get_actions()[1])
#     assert len(state_prob_cost) == 1
#     state4 = list(state_prob_cost.keys())[0]
#     assert state4.get_actions()[0].rtype == param.RobotType.Ground
#     assert state4.action_cost == 0.0
#     assert state4.uavs[0].need_action == False
#     assert state4.ugvs[0].need_action == True
#     assert state4.ugvs[1].need_action == True
#     assert len(state4.get_actions()) == len(state4.state_actions) == 1

#     # the 5th transition - assign action to the ugv 1
#     assert state4.get_actions()[0].rtype == param.RobotType.Ground
#     assert state4.get_actions()[0].target == 6
#     assert state4.cur_ugv_idx == 0
#     state_prob_cost = state4.transition(state4.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state5 = list(state_prob_cost.keys())[0]
#     assert state5.action_cost == 0.0
#     assert state5.uavs[0].need_action == False
#     assert state5.ugvs[0].need_action == False
#     assert state5.ugvs[1].need_action == True
#     assert len(state5.get_actions()) == len(state5.state_actions) == 1

#     # the 6th transition - assign action to the ugv 2, then move
#     assert state5.get_actions()[0].rtype == param.RobotType.Ground
#     assert state5.get_actions()[0].target == 6
#     assert state5.cur_ugv_idx == 1
#     state_prob_cost = state5.transition(state5.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state6 = list(state_prob_cost.keys())[0]
#     state6_b = list(state_prob_cost.keys())[1]
#     assert state6.action_cost > 0.0
#     assert state6.uavs[0].need_action == False
#     assert state6.ugvs[0].need_action == True
#     assert state6.ugvs[1].need_action == False
#     assert len(state6.get_actions()) == len(state6.state_actions) == 1

#     # the 7th transition - assign action to the ugv 1, then move
#     assert state6.get_actions()[0].rtype == param.RobotType.Ground
#     assert state6.get_actions()[0].target == 2
#     assert state6.cur_ugv_idx == 0
#     state_prob_cost = state6.transition(state6.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state7 = list(state_prob_cost.keys())[0]
#     assert state7.action_cost == 0.0
#     assert state7.uavs[0].need_action == False
#     assert state7.ugvs[0].need_action == False
#     assert state7.ugvs[1].need_action == True
#     assert len(state7.get_actions()) == len(state7.state_actions) == 1

#     # the 8th transition - assign action to the ugv 2, then move
#     assert state7.get_actions()[0].rtype == param.RobotType.Ground
#     assert state7.get_actions()[0].target == 2
#     assert state7.cur_ugv_idx == 1
#     state_prob_cost = state7.transition(state7.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state8 = list(state_prob_cost.keys())[0]
#     assert state8.action_cost > 0.0
#     assert state8.uavs[0].need_action == True
#     assert state8.ugvs[0].need_action == True
#     assert state8.ugvs[1].need_action == True
#     assert len(state8.get_actions()) == len(state8.state_actions) == 4


#     # check blocked state
#     assert state6_b.action_cost > 0.0
#     assert state6_b.uavs[0].need_action == False
#     assert state6_b.ugvs[0].need_action == True
#     assert state6_b.ugvs[1].need_action == False
#     assert len(state6_b.get_actions()) == len(state6_b.state_actions) == 1

#     # the 7th-block transition - assign action to the ugv 1, then move
#     assert state6_b.get_actions()[0].rtype == param.RobotType.Ground
#     assert state6_b.get_actions()[0].target == 1
#     assert state6_b.cur_ugv_idx == 0
#     state_prob_cost = state6_b.transition(state6_b.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state7_b = list(state_prob_cost.keys())[0]
#     assert state7_b.action_cost == 0.0
#     assert state7_b.uavs[0].need_action == False
#     assert state7_b.ugvs[0].need_action == False
#     assert state7_b.ugvs[1].need_action == True
#     assert len(state7_b.get_actions()) == len(state7_b.state_actions) == 1
#     assert state7_b.get_actions()[0].target == 1



# def test_stateDecPrior_transition_1uav2ugv_reachgoals():
#     print()
#     num_uav = 1
#     num_ugv = 2
#     starts, goals, graph = graphs.s_graph_2goals()
#     uavs = [Robot(position=[4.0, 4.0], cur_node=2, at_node=True, robot_type=param.RobotType.Drone) for i in range(num_uav)]
#     ugv1 = Robot(position=[4.0, 4.0], cur_node=2, at_node=True)
#     ugv2 = Robot(position=[3.0, 3.0], cur_node=6, at_node=False, edge=[2,6])
#     # ugv2.last_node = 1
#     ugvs = [ugv1, ugv2]

#     state = dec_prior.StateDecPrior(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=uavs)
#     # state.history.add_history(core.Action(rtype=param.RobotType.Drone, target=6), outcome=param.EventOutcome.TRAV)
#     assert all(uav.need_action == True for uav in state.uavs)
#     assert all(ugv.need_action == True for ugv in state.ugvs)
#     assert len(state.assigned_pois) == 0

#     # the first transition - assign action to the drone
#     assert state.get_actions()[0].rtype == param.RobotType.Drone
#     assert state.get_actions()[1].target == 7
#     state_prob_cost = state.transition(state.get_actions()[1])
#     assert len(state_prob_cost) == 1
#     state1 = list(state_prob_cost.keys())[0]
#     assert state1.action_cost == 0.0
#     # assert state1.heuristic == pytest.approx(8.0+8.243, 0.05)
#     assert state1.uavs[0].need_action == False
#     assert state1.ugvs[0].need_action == True
#     assert state1.ugvs[1].need_action == True
#     assert len(state1.get_actions()) == len(state1.state_actions) == 1

#     # the second transition - assign action to the ugv 1
#     assert state1.get_actions()[0].rtype == param.RobotType.Ground
#     assert state1.get_actions()[0].target == 9
#     state_prob_cost = state1.transition(state1.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state2 = list(state_prob_cost.keys())[0]
#     assert state2.action_cost == 0.0
#     # assert state2.heuristic == pytest.approx(8.0+8.243, 0.05)
#     assert state2.uavs[0].need_action == False
#     assert state2.ugvs[0].need_action == False
#     assert state2.ugvs[1].need_action == True
#     assert len(state2.get_actions()) == len(state2.state_actions) == 1
    
#     # the third transition - assign action to the ugv 2, then move
#     assert state2.get_actions()[0].rtype == param.RobotType.Ground
#     assert state2.get_actions()[0].target == 2
#     state_prob_cost = state2.transition(state2.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state3 = list(state_prob_cost.keys())[0]
#     assert state3.uavs[0].need_action == False
#     assert state3.ugvs[0].need_action == False
#     assert state3.ugvs[1].need_action == True
#     assert len(state3.get_actions()) == len(state3.state_actions) == 1
    
#     # the 4th transition - assign action to the ugv 2, then move
#     assert state3.get_actions()[0].rtype == param.RobotType.Ground
#     assert state3.get_actions()[0].target == 10
#     state_prob_cost = state3.transition(state3.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state4 = list(state_prob_cost.keys())[0]
#     assert state4.get_actions()[0].rtype == param.RobotType.Drone
#     assert state4.action_cost > 0.0
#     # assert state4.heuristic == pytest.approx(8.0+8.243, 0.05) # -1.885
#     assert state4.uavs[0].need_action == True
#     assert state4.ugvs[0].need_action == True
#     assert state4.ugvs[1].need_action == True
#     assert len(state4.get_actions()) == len(state4.state_actions) == 6
    
#     # the 5th transition - assign action to the uav
#     assert state4.get_actions()[0].rtype == param.RobotType.Drone
#     assert state4.get_actions()[0].target == 6
#     state_prob_cost = state4.transition(state4.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state5 = list(state_prob_cost.keys())[0]
#     assert state5.action_cost == 0.0
#     assert state5.uavs[0].need_action == False
#     assert state5.ugvs[0].need_action == True
#     assert state5.ugvs[1].need_action == True
#     assert len(state5.get_actions()) == len(state5.state_actions) == 1

#     # the 6th transition - assign action to the ugv 1
#     assert state5.get_actions()[0].rtype == param.RobotType.Ground
#     assert state5.get_actions()[0].target == 9
#     assert state5.cur_ugv_idx == 0
#     state_prob_cost = state5.transition(state5.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state6 = list(state_prob_cost.keys())[0]
#     assert state6.action_cost == 0.0
#     assert state6.uavs[0].need_action == False
#     assert state6.ugvs[0].need_action == False
#     assert state6.ugvs[1].need_action == True
#     assert len(state6.get_actions()) == len(state6.state_actions) == 1

#     # the 7th transition - assign action to the ugv 2
#     assert state6.get_actions()[0].rtype == param.RobotType.Ground
#     assert state6.get_actions()[0].target == 10
#     assert state6.cur_ugv_idx == 1
#     state_prob_cost = state6.transition(state6.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state7 = list(state_prob_cost.keys())[0]
#     assert state7.action_cost > 0.0
#     assert state7.uavs[0].need_action == True
#     assert state7.ugvs[0].need_action == True
#     assert state7.ugvs[1].need_action == True
#     assert len(state7.get_actions()) == len(state7.state_actions) == 5

#     # the 8th transition - assign action to the uav
#     assert state7.get_actions()[0].rtype == param.RobotType.Drone
#     assert state7.get_actions()[0].target == 8
#     assert state7.cur_ugv_idx == -1
#     state_prob_cost = state7.transition(state7.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state8 = list(state_prob_cost.keys())[0]
#     assert state8.action_cost == 0.0
#     assert state8.uavs[0].need_action == False
#     assert state8.ugvs[0].need_action == True
#     assert state8.ugvs[1].need_action == True
#     assert len(state8.get_actions()) == len(state8.state_actions) == 1

#     # the 9th transition - assign action to the ugv 1
#     assert state8.get_actions()[0].rtype == param.RobotType.Ground
#     assert state8.get_actions()[0].target == 9
#     assert state8.cur_ugv_idx == 0
#     state_prob_cost = state8.transition(state8.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state9 = list(state_prob_cost.keys())[0]
#     assert state9.action_cost == 0.0
#     assert state9.uavs[0].need_action == False
#     assert state9.ugvs[0].need_action == False
#     assert state9.ugvs[1].need_action == True
#     assert len(state9.get_actions()) == len(state9.state_actions) == 1

#     # the 10th transition - assign action to the ugv 2
#     assert state9.get_actions()[0].rtype == param.RobotType.Ground
#     assert state9.get_actions()[0].target == 10
#     assert state9.cur_ugv_idx == 1
#     state_prob_cost = state9.transition(state9.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state10 = list(state_prob_cost.keys())[0]
#     assert state10.action_cost > 0.0
#     assert state10.uavs[0].need_action == True
#     assert state10.ugvs[0].need_action == True
#     assert state10.ugvs[1].need_action == True
#     assert len(state10.get_actions()) == len(state10.state_actions) == 4

# # the 11th transition - assign action to the uav
#     assert state10.get_actions()[0].rtype == param.RobotType.Drone
#     assert state10.get_actions()[0].target == 9
#     assert state10.cur_ugv_idx == -1
#     state_prob_cost = state10.transition(state10.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state11 = list(state_prob_cost.keys())[0]
#     assert state11.action_cost == 0.0
#     assert state11.uavs[0].need_action == False
#     assert state11.ugvs[0].need_action == True
#     assert state11.ugvs[1].need_action == True
#     assert len(state11.get_actions()) == len(state11.state_actions) == 1

# # the 12th transition - assign action to the ugv 1
#     assert state11.get_actions()[0].rtype == param.RobotType.Ground
#     assert state11.get_actions()[0].target == 9
#     assert state11.cur_ugv_idx == 0
#     state_prob_cost = state11.transition(state11.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state12 = list(state_prob_cost.keys())[0]
#     assert state12.action_cost == 0.0
#     assert state12.uavs[0].need_action == False
#     assert state12.ugvs[0].need_action == False
#     assert state12.ugvs[1].need_action == True
#     assert len(state12.get_actions()) == len(state12.state_actions) == 1

# # the 13th transition - assign action to the ugv 2, then move
#     assert state12.get_actions()[0].rtype == param.RobotType.Ground
#     assert state12.get_actions()[0].target == 10
#     assert state12.cur_ugv_idx == 1
#     state_prob_cost = state12.transition(state12.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state13 = list(state_prob_cost.keys())[0]
#     assert state13.action_cost > 0.0
#     assert state13.uavs[0].need_action == True
#     assert state13.ugvs[0].need_action == True
#     assert state13.ugvs[1].need_action == True
#     assert len(state13.get_actions()) == len(state13.state_actions) == 1

# # the 14th transition - assign action to the ugv 1
#     assert state13.get_actions()[0].rtype == param.RobotType.Ground
#     assert state13.get_actions()[0].target == 4
#     assert state13.cur_ugv_idx == 0
#     state_prob_cost = state13.transition(state13.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state14 = list(state_prob_cost.keys())[0]
#     assert state14.action_cost == 0.0
#     assert state14.uavs[0].need_action == True
#     assert state14.ugvs[0].need_action == False
#     assert state14.ugvs[1].need_action == True
#     assert len(state14.get_actions()) == len(state14.state_actions) == 3

# # the 15th transition - assign action to the uav
#     assert state14.get_actions()[0].rtype == param.RobotType.Drone
#     assert state14.get_actions()[0].target == 10
#     assert state14.cur_ugv_idx == -1
#     state_prob_cost = state14.transition(state14.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state15 = list(state_prob_cost.keys())[0]
#     assert state15.action_cost == 0.0
#     assert state15.uavs[0].need_action == False
#     assert state15.ugvs[0].need_action == False
#     assert state15.ugvs[1].need_action == True
#     assert len(state15.get_actions()) == len(state15.state_actions) == 1

# # the 16th transition - assign action to the ugv2, then move
#     assert state15.get_actions()[0].rtype == param.RobotType.Ground
#     assert state15.get_actions()[0].target == 10
#     assert state15.cur_ugv_idx == 1
#     state_prob_cost = state15.transition(state15.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state16 = list(state_prob_cost.keys())[0]
#     assert state16.action_cost > 0.0
#     assert state16.uavs[0].need_action == True
#     assert state16.ugvs[0].need_action == True
#     assert state16.ugvs[1].need_action == True
#     assert len(state16.get_actions()) == len(state16.state_actions) == 1

# # the 17th transition - assign action to the ugv2
#     assert state16.get_actions()[0].rtype == param.RobotType.Ground
#     assert state16.get_actions()[0].target == 5
#     assert state16.cur_ugv_idx == 1
#     state_prob_cost = state16.transition(state16.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state17 = list(state_prob_cost.keys())[0]
#     assert state17.action_cost == 0.0
#     assert state17.uavs[0].need_action == True
#     assert state17.ugvs[0].need_action == True
#     assert state17.ugvs[1].need_action == False
#     assert len(state17.get_actions()) == len(state17.state_actions) == 2

# # the 18th transition - assign action to the uav
#     assert state17.get_actions()[0].rtype == param.RobotType.Drone
#     assert state17.get_actions()[0].target == 11
#     assert state17.cur_ugv_idx == -1
#     state_prob_cost = state17.transition(state17.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state18 = list(state_prob_cost.keys())[0]
#     assert state18.action_cost == 0.0
#     assert state18.uavs[0].need_action == False
#     assert state18.ugvs[0].need_action == True
#     assert state18.ugvs[1].need_action == False
#     assert len(state18.get_actions()) == len(state18.state_actions) == 1

# # the 19th transition - assign action to the ugv 1, then move
#     assert state18.get_actions()[0].rtype == param.RobotType.Ground
#     assert state18.get_actions()[0].target == 4
#     assert state18.cur_ugv_idx == 0
#     state_prob_cost = state18.transition(state18.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state19 = list(state_prob_cost.keys())[0]
#     assert state19.action_cost > 0.0
#     assert state19.uavs[0].need_action == True
#     assert state19.ugvs[0].need_action == True
#     assert state19.ugvs[1].need_action == True
#     assert len(state19.get_actions()) == len(state19.state_actions) == 1

# # the 20th transition - assign action to the uav
#     assert state19.get_actions()[0].rtype == param.RobotType.Drone
#     assert state19.get_actions()[0].target == 12
#     assert state19.cur_ugv_idx == -1
#     state_prob_cost = state19.transition(state19.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state20 = list(state_prob_cost.keys())[0]
#     assert state20.action_cost == 0.0
#     assert state20.uavs[0].need_action == False
#     assert state20.ugvs[0].need_action == True
#     assert state20.ugvs[1].need_action == True
#     assert len(state20.get_actions()) == len(state20.state_actions) == 1

# # the 21th transition - assign action to the ugv 1
#     assert state20.get_actions()[0].rtype == param.RobotType.Ground
#     assert state20.get_actions()[0].target == 4
#     assert state20.cur_ugv_idx == 0
#     state_prob_cost = state20.transition(state20.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state21 = list(state_prob_cost.keys())[0]
#     assert state21.action_cost == 0.0
#     assert state21.uavs[0].need_action == False
#     assert state21.ugvs[0].need_action == False
#     assert state21.ugvs[1].need_action == True
#     assert len(state21.get_actions()) == len(state21.state_actions) == 1    

# # the 22th transition - assign action to the ugv 2, then move
#     assert state21.get_actions()[0].rtype == param.RobotType.Ground
#     assert state21.get_actions()[0].target == 5
#     assert state21.cur_ugv_idx == 1
#     state_prob_cost = state21.transition(state21.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state22 = list(state_prob_cost.keys())[0]
#     assert state22.action_cost > 0.0
#     assert state22.uavs[0].need_action == False
#     assert state22.ugvs[0].need_action == False
#     assert state22.ugvs[1].need_action == True
#     assert len(state22.get_actions()) == len(state22.state_actions) == 1

# # the 23th transition - assign action to the ugv 2, then move
#     assert state22.get_actions()[0].rtype == param.RobotType.Ground
#     assert state22.get_actions()[0].target == 5
#     assert state22.cur_ugv_idx == 1
#     state_prob_cost = state22.transition(state22.get_actions()[0])
#     assert len(state_prob_cost) == 2
#     state23 = list(state_prob_cost.keys())[0]
#     assert state23.action_cost > 0.0
#     assert state23.uavs[0].need_action == True
#     assert state23.ugvs[0].need_action == True
#     assert state23.ugvs[1].need_action == False
#     assert len(state23.get_actions()) == len(state23.state_actions) == 1

# # the 24th transition - assign action to the uav
#     assert state23.get_actions()[0].rtype == param.RobotType.Drone
#     assert state23.get_actions()[0].target == 4
#     assert state23.cur_ugv_idx == -1
#     state_prob_cost = state23.transition(state23.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state24 = list(state_prob_cost.keys())[0]
#     assert state24.action_cost == 0.0
#     assert state24.uavs[0].need_action == False
#     assert state24.ugvs[0].need_action == True
#     assert state24.ugvs[1].need_action == False
#     assert len(state24.get_actions()) == len(state24.state_actions) == 1

# # the 25th transition - assign action to the ugv 1
#     assert state24.get_actions()[0].rtype == param.RobotType.Ground
#     assert state24.get_actions()[0].target == 4
#     assert state24.cur_ugv_idx == 0
#     state_prob_cost = state24.transition(state24.get_actions()[0])
#     assert len(state_prob_cost) == 1
#     state25 = list(state_prob_cost.keys())[0]
#     assert state25.action_cost > 0.0
#     assert state25.uavs[0].need_action == False
#     assert state25.ugvs[0].need_action == True
#     assert state25.ugvs[1].need_action == False
#     assert len(state25.get_actions()) == len(state25.state_actions) == 1
#     assert state25.is_goal_state == True

    
# def test_stateDecPrior_transition_1uav2ugv_2goals():
    # print()
    # num_uav = 1
    # num_ugv = 2
    # starts, goals, graph = graphs.s_graph_2goals()
    # uavs = [Robot(position=[0.0, 0.0], cur_node=starts[i].id, robot_type=param.RobotType.Drone) for i in range(num_uav)]
    # ugv1 = Robot(position=[0.0, 0.0], cur_node=starts[0].id)
    # ugv2 = Robot(position=[1.0, 1.0], cur_node=starts[0].id, at_node=False, edge=[1,6])
    # # ugv2.last_node = 1
    # ugvs = [ugv1, ugv2]

    # state = dec_prior.StateDecPrior(graph=graph, goalIDs=[g.id for g in goals], ugvs=ugvs, drones=uavs)
    # assert state.heuristic == pytest.approx(8.0+8.243, 0.05)
    # assert state.history.get_data_length() == len(graph.vertices)
    # assert len(state.state_actions) == len(state.get_actions()) == 7
    # assert all(uav.need_action == True for uav in state.uavs)
    # assert all(ugv.need_action == True for ugv in state.ugvs)
    # assert len(state.assigned_pois) == 0

    # # the first transition - assign action to the drone
    # assert state.get_actions()[0].rtype == param.RobotType.Drone
    # assert state.get_actions()[1].target == 7
    # state_prob_cost = state.transition(state.get_actions()[1])
    # assert len(state_prob_cost) == 1
    # state1 = list(state_prob_cost.keys())[0]
    # assert state1.action_cost == 0.0
    # assert state1.heuristic == pytest.approx(8.0+8.243, 0.05)
    # assert state1.uavs[0].need_action == False
    # assert state1.ugvs[0].need_action == True
    # assert state1.ugvs[1].need_action == True
    # assert len(state1.get_actions()) == len(state1.state_actions) == 1

    # # the second transition - assign action to the ugv 1
    # assert state1.get_actions()[0].rtype == param.RobotType.Ground
    # assert state1.get_actions()[0].target == 6
    # state_prob_cost = state1.transition(state1.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state2 = list(state_prob_cost.keys())[0]
    # assert state2.action_cost == 0.0
    # assert state2.heuristic == pytest.approx(8.0+8.243, 0.05)
    # assert state2.uavs[0].need_action == False
    # assert state2.ugvs[0].need_action == False
    # assert state2.ugvs[1].need_action == True
    # assert len(state2.get_actions()) == len(state2.state_actions) == 1
    
    # # the third transition - assign action to the ugv 2, then move
    # assert state2.get_actions()[0].rtype == param.RobotType.Ground
    # assert state2.get_actions()[0].target == 6
    # state_prob_cost = state2.transition(state2.get_actions()[0])
    # assert len(state_prob_cost) == 2
    # state3_p = list(state_prob_cost.keys())[0]
    # assert state3_p.action_cost == pytest.approx(2.0/param.VEL_RATIO, 0.05)
    # assert state3_p.heuristic == pytest.approx(8.0+8.243, 0.05) # -1.885
    # assert state3_p.uavs[0].need_action == True
    # assert state3_p.ugvs[0].need_action == True
    # assert state3_p.ugvs[1].need_action == True
    # assert len(state3_p.get_actions()) == len(state3_p.state_actions) == 6
    
    # # the 4th transition - assign action to the uav
    # assert state3_p.get_actions()[0].rtype == param.RobotType.Drone
    # assert state3_p.get_actions()[1].target == 8
    # state_prob_cost = state3_p.transition(state3_p.get_actions()[1])
    # assert len(state_prob_cost) == 1
    # state4 = list(state_prob_cost.keys())[0]
    # assert state4.get_actions()[0].rtype == param.RobotType.Ground
    # assert state4.action_cost == 0.0
    # assert state4.heuristic == pytest.approx(8.0+8.243, 0.05) # -1.885
    # assert state4.uavs[0].need_action == False
    # assert state4.ugvs[0].need_action == True
    # assert state4.ugvs[1].need_action == True
    # assert len(state4.get_actions()) == len(state4.state_actions) == 1
    
    # # the 5th transition - assign action to the ugv 1
    # assert state4.get_actions()[0].rtype == param.RobotType.Ground
    # assert state4.get_actions()[0].target == 6
    # state_prob_cost = state4.transition(state4.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state5 = list(state_prob_cost.keys())[0]
    # assert state5.action_cost == 0.0
    # assert state5.heuristic == pytest.approx(8.0+8.243, 0.05) # -1.885
    # assert state5.uavs[0].need_action == False
    # assert state5.ugvs[0].need_action == False
    # assert state5.ugvs[1].need_action == True
    # assert len(state5.get_actions()) == len(state5.state_actions) == 1
    
    # # the 6th transition - assign action to the ugv 2, then move
    # assert state5.get_actions()[0].rtype == param.RobotType.Ground
    # assert state5.get_actions()[0].target == 6
    # state_prob_cost = state5.transition(state5.get_actions()[0])
    # assert len(state_prob_cost) == 2
    # state6 = list(state_prob_cost.keys())[0]
    # assert state6.action_cost > 0.0
    # assert state6.heuristic == pytest.approx(8.0+8.243, 0.05) # -1.885
    # assert state6.uavs[0].need_action == False
    # assert state6.ugvs[0].need_action == True
    # assert state6.ugvs[1].need_action == True
    # assert len(state6.get_actions()) == len(state6.state_actions) == 1 
    
    # # the 7th transition - reassign action to the ugv 2
    # assert state6.get_actions()[0].rtype == param.RobotType.Ground
    # assert state6.cur_ugv_idx == 1
    # assert state6.get_actions()[0].target == 2
    # state_prob_cost = state6.transition(state6.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state7 = list(state_prob_cost.keys())[0]
    # assert state7.get_actions()[0].rtype == param.RobotType.Ground
    # assert state7.action_cost == 0.0
    # assert state7.heuristic == pytest.approx(8.0+8.243, 0.05) # -1.885
    # assert state7.uavs[0].need_action == False
    # assert state7.ugvs[0].need_action == True
    # assert state7.ugvs[1].need_action == False
    # assert len(state7.get_actions()) == len(state7.state_actions) == 1
    
    # # the 8th transition - reassign action to the ugv 1, then move
    # assert state7.get_actions()[0].rtype == param.RobotType.Ground
    # assert state7.cur_ugv_idx == 0
    # assert state7.get_actions()[0].target == 6
    # state_prob_cost = state7.transition(state7.get_actions()[0])
    # assert len(state_prob_cost) == 2
    # state8_p = list(state_prob_cost.keys())[0]
    # assert state8_p.get_actions()[0].rtype == param.RobotType.Drone
    # assert state8_p.action_cost > 0.0
    # assert state8_p.heuristic == pytest.approx(8.0+8.243, 0.05) # -1.885
    # assert state8_p.uavs[0].need_action == True
    # assert state8_p.ugvs[0].need_action == True
    # assert state8_p.ugvs[1].need_action == True
    # assert len(state8_p.get_actions()) == len(state8_p.state_actions) == 4
    
    # # the 9th transition - assign action to the uav
    # assert state8_p.get_actions()[0].rtype == param.RobotType.Drone
    # assert state8_p.cur_ugv_idx == -1
    # assert state8_p.get_actions()[0].target == 9
    # state_prob_cost = state8_p.transition(state8_p.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state9 = list(state_prob_cost.keys())[0]
    # assert state9.get_actions()[0].rtype == param.RobotType.Ground
    # assert state9.action_cost == 0.0
    # assert state9.heuristic == pytest.approx(8.0+8.243, 0.05) # -1.885
    # assert state9.uavs[0].need_action == False
    # assert state9.ugvs[0].need_action == True
    # assert state9.ugvs[1].need_action == True
    # assert len(state9.get_actions()) == len(state9.state_actions) == 1
    
    # # the 10th transition - reassign action to the first ugv
    # assert state9.get_actions()[0].rtype == param.RobotType.Ground
    # assert state9.cur_ugv_idx == 0
    # assert state9.get_actions()[0].target == 6
    # state_prob_cost = state9.transition(state9.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state10 = list(state_prob_cost.keys())[0]
    # assert state10.action_cost == 0.0
    # assert state10.heuristic == pytest.approx(8.0+8.243, 0.05) # -1.885
    # assert state10.uavs[0].need_action == False
    # assert state10.ugvs[0].need_action == False
    # assert state10.ugvs[1].need_action == True
    # assert len(state10.get_actions()) == len(state10.state_actions) == 1
    
    # # the 11th transition - reassign action to ugv 2
    # assert state10.get_actions()[0].rtype == param.RobotType.Ground
    # assert state10.cur_ugv_idx == 1
    # assert state10.get_actions()[0].target == 2
    # state_prob_cost = state10.transition(state10.get_actions()[0])
    # assert len(state_prob_cost) == 2
    # state11 = list(state_prob_cost.keys())[0]
    # assert state11.get_actions()[0].rtype == param.RobotType.Drone
    # assert state11.action_cost > 0.0
    # # assert state11.heuristic == pytest.approx(8.0+8.243, 0.05) # -1.885
    # assert state11.uavs[0].need_action == True
    # assert state11.ugvs[0].need_action == True
    # assert state11.ugvs[1].need_action == True
    # assert len(state11.get_actions()) == len(state11.state_actions) == 3
    
    # # the 12th transition - assign action to uav
    # assert state11.get_actions()[0].rtype == param.RobotType.Drone
    # assert state11.cur_ugv_idx == -1
    # assert state11.get_actions()[0].target == 10
    # state_prob_cost = state11.transition(state11.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state12 = list(state_prob_cost.keys())[0]
    # assert state12.get_actions()[0].rtype == param.RobotType.Ground
    # assert state12.action_cost == 0.0
    # assert state12.uavs[0].need_action == False
    # assert state12.ugvs[0].need_action == True
    # assert state12.ugvs[1].need_action == True
    # assert len(state12.get_actions()) == len(state12.state_actions) == 1

    # # the 13th transition - reassign action to ugv 1
    # assert state12.get_actions()[0].rtype == param.RobotType.Ground
    # assert state12.cur_ugv_idx == 0
    # assert state12.get_actions()[0].target == 6
    # state_prob_cost = state12.transition(state12.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state13 = list(state_prob_cost.keys())[0]
    # assert state13.get_actions()[0].rtype == param.RobotType.Ground
    # assert state13.action_cost == 0.0
    # assert state13.uavs[0].need_action == False
    # assert state13.ugvs[0].need_action == False
    # assert state13.ugvs[1].need_action == True
    # assert len(state13.get_actions()) == len(state13.state_actions) == 1
    
    # # the 14th transition - reassign action to ugv 2, then move
    # assert state13.get_actions()[0].rtype == param.RobotType.Ground
    # assert state13.cur_ugv_idx == 1
    # assert state13.get_actions()[0].target == 2
    # state_prob_cost = state13.transition(state13.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state14 = list(state_prob_cost.keys())[0]
    # assert state14.get_actions()[0].rtype == param.RobotType.Ground
    # assert state14.action_cost > 0.0
    # assert state14.uavs[0].need_action == False
    # assert state14.ugvs[0].need_action == True
    # assert state14.ugvs[1].need_action == False
    # assert len(state14.get_actions()) == len(state14.state_actions) == 1

    # # the 15th transition - assign action to ugv 1, then move
    # assert state14.get_actions()[0].rtype == param.RobotType.Ground
    # assert state14.cur_ugv_idx == 0
    # # print(f"Current node: {state14.ugvs[0].last_node} and current pose: {state14.ugvs[0].cur_pose}")
    # assert state14.get_actions()[0].target == 2
    # state_prob_cost = state14.transition(state14.get_actions()[0])
    # assert len(state_prob_cost) == 2
    # state15 = list(state_prob_cost.keys())[0]
    # assert state15.get_actions()[0].rtype == param.RobotType.Drone
    # assert state15.action_cost > 0.0
    # assert state15.uavs[0].need_action == True
    # assert state15.ugvs[0].need_action == True
    # assert state15.ugvs[1].need_action == True
    # assert len(state15.get_actions()) == len(state15.state_actions) == 2

    # # the 16th transition - assign action to uav
    # assert state15.get_actions()[0].rtype == param.RobotType.Drone
    # assert state15.cur_ugv_idx == -1
    # assert state15.get_actions()[1].target == 12
    # state_prob_cost = state15.transition(state15.get_actions()[1])
    # assert len(state_prob_cost) == 1
    # state16 = list(state_prob_cost.keys())[0]
    # assert state16.action_cost == 0.0
    # assert state16.uavs[0].need_action == False
    # assert state16.ugvs[0].need_action == True
    # assert state16.ugvs[1].need_action == True
    # assert len(state16.get_actions()) == len(state16.state_actions) == 1
    
    # # the 17th transition - assign action to ugv 1
    # assert state16.get_actions()[0].rtype == param.RobotType.Ground
    # assert state16.cur_ugv_idx == 0
    # assert state16.get_actions()[0].target == 2
    # state_prob_cost = state16.transition(state16.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state17 = list(state_prob_cost.keys())[0]
    # assert state17.action_cost == 0.0
    # assert state17.uavs[0].need_action == False
    # assert state17.ugvs[0].need_action == False
    # assert state17.ugvs[1].need_action == True
    # assert len(state17.get_actions()) == len(state17.state_actions) == 1
    
    # # the 18th transition - assign action to ugv 2
    # assert state17.get_actions()[0].rtype == param.RobotType.Ground
    # assert state17.cur_ugv_idx == 1
    # assert state17.get_actions()[0].target == 2
    # state_prob_cost = state17.transition(state17.get_actions()[0])
    # assert len(state_prob_cost) == 2
    # state18 = list(state_prob_cost.keys())[0]
    # assert state18.action_cost > 0.0
    # assert state18.uavs[0].need_action == True
    # assert state18.ugvs[0].need_action == True
    # assert state18.ugvs[1].need_action == True
    # assert len(state18.get_actions()) == len(state18.state_actions) == 1
    
    # # the 19th transition - assign action to uav
    # assert state18.get_actions()[0].rtype == param.RobotType.Drone
    # assert state18.cur_ugv_idx == -1
    # assert state18.get_actions()[0].target == 11
    # state_prob_cost = state18.transition(state18.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state19 = list(state_prob_cost.keys())[0]
    # assert state19.action_cost == 0.0
    # assert state19.uavs[0].need_action == False
    # assert state19.ugvs[0].need_action == True
    # assert state19.ugvs[1].need_action == True
    # assert len(state19.get_actions()) == len(state19.state_actions) == 1

    # # the 20th transition - assign action to ugv 1
    # assert state19.get_actions()[0].rtype == param.RobotType.Ground
    # assert state19.cur_ugv_idx == 0
    # assert state19.get_actions()[0].target == 2
    # state_prob_cost = state19.transition(state19.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state20 = list(state_prob_cost.keys())[0]
    # assert state20.action_cost == 0.0
    # assert state20.uavs[0].need_action == False
    # assert state20.ugvs[0].need_action == False
    # assert state20.ugvs[1].need_action == True
    # assert len(state20.get_actions()) == len(state20.state_actions) == 1
    
    # # the 21th transition - assign action to ugv 2
    # assert state20.get_actions()[0].rtype == param.RobotType.Ground
    # assert state20.cur_ugv_idx == 1
    # assert state20.get_actions()[0].target == 2
    # state_prob_cost = state20.transition(state20.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state21 = list(state_prob_cost.keys())[0]
    # assert state21.action_cost > 0.0
    # assert state21.uavs[0].need_action == False
    # assert state21.ugvs[0].need_action == False
    # assert state21.ugvs[1].need_action == True
    # assert len(state21.get_actions()) == len(state21.state_actions) == 1
    
    # # the 22th transition - assign action to ugv 2, then move
    # assert state21.get_actions()[0].rtype == param.RobotType.Ground
    # assert state21.cur_ugv_idx == 1
    # assert state21.get_actions()[0].target == 10
    # state_prob_cost = state21.transition(state21.get_actions()[0])
    # assert len(state_prob_cost) == 2
    # state22_b = list(state_prob_cost.keys())[1]
    # assert state22_b.action_cost > 0.0
    # assert state22_b.uavs[0].need_action == True
    # assert state22_b.ugvs[0].need_action == True
    # assert state22_b.ugvs[1].need_action == True
    # assert len(state22_b.get_actions()) == len(state22_b.state_actions) == 1
    
    # # the 23th transition - assign action to the uav
    # assert state22_b.get_actions()[0].rtype == param.RobotType.Drone
    # assert state22_b.cur_ugv_idx == -1
    # assert state22_b.get_actions()[0].target == 4
    # state_prob_cost = state22_b.transition(state22_b.get_actions()[0])
    # assert len(state_prob_cost) == 1
    # state23 = list(state_prob_cost.keys())[0]
    # assert state23.action_cost == 0.0
    # assert state23.uavs[0].need_action == False
    # assert state23.ugvs[0].need_action == True
    # assert state23.ugvs[1].need_action == True
    # assert len(state23.get_actions()) == len(state23.state_actions) == 1