import pytest
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.robot import Robot
from sctp import param, dstate_dec
from sctp.utils import plotting, paths
import matplotlib.pyplot as plt

def test_dronestate_transition_lgraph():
    print()
    start, goal, l_graph = graphs.linear_graph_unc()
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=param.RobotType.Drone) for _ in range(1)]
    actions = [core.Action(target=v.id, rtype=param.RobotType.Drone) for v in l_graph.pois if v.id != start.id]
    robot_edges = [[1,1]]
    robot_pos = [[0.0,0.0]]
    state = dstate_dec.DronesState(actions=actions, robot_pos=robot_pos, redges=robot_edges, \
                            graph=l_graph, restID=goal.id, drones=drones)
    # assert state.history.get_data_length() == len(l_graph.vertices)
    assert len(state.actions) == len(state.get_actions()) == 2
    assert all(uav.need_action == True for uav in state.uavs)
    assert len(state.assigned_pois) == 0

    state2 = state.copy()
    assert state2 != state
    # assert state2.history == state.history
    for i in range(len(state2.uavs)):
        assert state2.uavs[i] != state.uavs[i]
    assert state2.actions == state.actions
    assert state2.assigned_pois == state.assigned_pois

    # the first transition - assign action to the drone
    assert len(state.get_actions()) == 2
    # print(state.get_actions()[1].start_pose)    
    state_prob_cost = state.transition(state.get_actions()[0])
    
    print(f"{state.get_actions()[0].target}, {state.get_actions()[0].start_pose}")
    print(f"{state.get_actions()[1].target}, {state.get_actions()[1].start_pose}")
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    print(f"The cost of action is {state1.action_cost}")
    print(f"{state1.get_actions()[0].target} {state1.get_actions()[0].start_pose}")
    assert state1.uavs[0].cur_pose[0] == 2.5 and state1.uavs[0].cur_pose[1] ==0.0
    assert state1.action_cost == state_prob_cost[state1][1]
    assert state1.action_cost == 2.5/param.VEL_RATIO
    assert state1.uavs[0].need_action == True


def test_dronestate_transition_twodrones_lg():
    start, goal, l_graph = graphs.linear_graph_unc()
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=param.RobotType.Drone) for _ in range(2)]
    actions = [core.Action(target=v.id, rtype=param.RobotType.Drone) for v in l_graph.pois if v.id != start.id]
    robot_edges = [[1,1]]
    robot_pos = [[0.0,0.0]]
    state = dstate_dec.DronesState(actions=actions, robot_pos=robot_pos, redges=robot_edges, \
                            graph=l_graph, restID=goal.id, drones=drones)
    # assert state.history.get_data_length() == len(l_graph.vertices)
    assert len(state.actions) == len(state.get_actions()) == 2
    assert all(uav.need_action == True for uav in state.uavs)
    assert len(state.assigned_pois) == 0

    state2 = state.copy()
    assert state2 != state
    # assert state2.history == state.history
    for i in range(len(state2.uavs)):
        assert state2.uavs[i] != state.uavs[i]
    assert state2.actions == state.actions
    assert state2.assigned_pois == state.assigned_pois

    # # the first transition - assign action to the drone
    assert len(state.get_actions()) == 2
    state_prob_cost = state.transition(state.get_actions()[0])
    assert state.get_actions()[0].target == 4
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    assert state1.uavs[1].need_action == True
    assert state1.uavs[0].need_action == False
    assert state1.uavs[0].cur_pose[0] == 0.0 and state1.uavs[0].cur_pose[1] ==0.0
    assert state1.action_cost == 0.0
    assert len(state1.get_actions()) == 1
    assert state1.get_actions()[0].target == 5
    assert len(state1.assigned_pois) == 1
    
    ## second transition - assign action to the second drone
    state_prob_cost = state1.transition(state1.get_actions()[0])
    assert len(state_prob_cost) == 1
    state2 = list(state_prob_cost.keys())[0]
    assert state2.uavs[0].need_action == True 
    assert state2.uavs[1].need_action == False
    assert state2.uavs[1].cur_pose[0] == 2.5 and state2.uavs[1].cur_pose[1] ==0.0
    assert state2.uavs[0].cur_pose[0] == 2.5 and state2.uavs[0].cur_pose[1] ==0.0
    assert state2.action_cost == 2.5/param.VEL_RATIO
    assert state2.is_goal_state == False
    
    ## third transition - assign action to the first drone
    assert len(state2.get_actions()) == 1
    assert state2.get_actions()[0].target == 3
    state_prob_cost = state2.transition(state2.get_actions()[0])
    assert len(state_prob_cost) == 1
    state3 = list(state_prob_cost.keys())[0]
    assert state3.uavs[1].need_action == True 
    assert state3.uavs[0].need_action == False
    assert state3.uavs[1].cur_pose[0] == 10.0 and state3.uavs[1].cur_pose[1] ==0.0
    assert state3.uavs[0].cur_pose[0] == 10.0 and state3.uavs[0].cur_pose[1] ==0.0
    assert state3.action_cost == 7.5/param.VEL_RATIO
    assert state3.is_goal_state == False
    
    ## forth transition - assign action to the first drone
    assert len(state3.get_actions()) == 1
    assert state3.get_actions()[0].target == 3
    state_prob_cost = state3.transition(state3.get_actions()[0])
    assert len(state_prob_cost) == 1
    state4 = list(state_prob_cost.keys())[0]
    assert state4.uavs[1].need_action == True 
    assert state4.uavs[0].need_action == False
    assert state4.uavs[1].cur_pose[0] == 15.0 and state4.uavs[1].cur_pose[1] ==0.0
    assert state4.uavs[0].cur_pose[0] == 15.0 and state4.uavs[0].cur_pose[1] ==0.0
    assert state4.action_cost == 5.0/param.VEL_RATIO
    assert state4.is_goal_state == True


def test_dronestate_transition_twodrones_dg():
    start, goal, graph = graphs.disjoint_unc()
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=param.RobotType.Drone) for _ in range(2)]
    actions = [core.Action(target=v.id, rtype=param.RobotType.Drone) for v in graph.pois if v.id != start.id]
    robot_edges = [[1,1]]
    robot_pos = [[0.0,0.0]]
    state = dstate_dec.DronesState(actions=actions, robot_pos=robot_pos, redges=robot_edges, \
                            graph=graph, restID=goal.id, drones=drones)
    assert len(state.actions) == len(state.get_actions()) == 4
    assert all(uav.need_action == True for uav in state.uavs)
    assert len(state.assigned_pois) == 0

    state2 = state.copy()
    assert state2 != state
    for i in range(len(state2.uavs)):
        assert state2.uavs[i] != state.uavs[i]
    assert state2.actions == state.actions
    assert state2.assigned_pois == state.assigned_pois

    # # the first transition - assign action to the drone
    state_prob_cost = state.transition(state.get_actions()[0])
    assert state.get_actions()[0].target == 5
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    assert state1.uavs[0].need_action == False
    assert state1.uavs[1].need_action == True
    assert state1.uavs[0].cur_pose[0] == 0.0 and state1.uavs[0].cur_pose[1] ==0.0
    assert state1.action_cost == 0.0
    assert len(state1.get_actions()) == 3
    assert len(state1.assigned_pois) == 1
    
    # # the second transition - assign action to the drone
    state_prob_cost = state1.transition(state1.get_actions()[0])
    assert state1.get_actions()[0].target == 6
    assert len(state_prob_cost) == 1
    state2 = list(state_prob_cost.keys())[0]
    assert state2.uavs[0].need_action == True
    assert state2.uavs[1].need_action == False
    assert state2.uavs[0].cur_pose[0] == 2.0 and state2.uavs[0].cur_pose[1] ==0.0
    assert state2.action_cost == 2.0/param.VEL_RATIO
    assert len(state2.get_actions()) == 2
    assert len(state2.assigned_pois) == 2

    # # the third transition - assign action to the drone
    state_prob_cost = state2.transition(state2.get_actions()[1])
    assert state2.get_actions()[1].target == 8
    assert len(state_prob_cost) == 1
    state3 = list(state_prob_cost.keys())[0]
    assert state3.uavs[0].need_action == True
    assert state3.uavs[1].need_action == False
    assert state3.uavs[0].cur_pose[0] == 2.0 and state3.uavs[0].cur_pose[1] ==2.0
    assert state3.action_cost == 2.0/param.VEL_RATIO
    assert len(state3.get_actions()) == 1
    assert len(state3.assigned_pois) == 3

    # # the forth transition - assign action to the drone
    assert state3.get_actions()[0].target == 7
    state_prob_cost = state3.transition(state3.get_actions()[0])
    assert len(state_prob_cost) == 1
    state4 = list(state_prob_cost.keys())[0]
    assert state4.uavs[0].need_action == False
    assert state4.uavs[1].need_action == True
    assert state4.uavs[1].cur_pose[0] == 6.0 and state4.uavs[1].cur_pose[1] == pytest.approx(2.0,0.05)
    assert state4.action_cost == pytest.approx((6.325-4.0)/param.VEL_RATIO, 0.05)
    assert len(state4.get_actions()) == 1
    assert len(state4.assigned_pois) == 4
    assert state4.is_goal_state == False
    
    # # the forth transition - assign action to the drone
    assert state4.get_actions()[0].target == 3
    state_prob_cost = state4.transition(state4.get_actions()[0])
    assert len(state_prob_cost) == 1
    state5 = list(state_prob_cost.keys())[0]
    assert state5.uavs[0].need_action == True
    assert state5.uavs[1].need_action == False
    assert state5.uavs[0].cur_pose[0] == pytest.approx(6.0, 0.05) and state5.uavs[0].cur_pose[1] == pytest.approx(0.0,0.05)
    assert state5.action_cost == pytest.approx((8.472-6.325)/param.VEL_RATIO, 0.05)
    assert len(state5.get_actions()) == 1
    assert len(state5.assigned_pois) == 4
    assert state5.is_goal_state == False

    # # the five transition - assign action to the drone
    assert state5.get_actions()[0].target == 3
    state_prob_cost = state5.transition(state5.get_actions()[0])
    assert len(state_prob_cost) == 1
    state6 = list(state_prob_cost.keys())[0]
    assert state6.uavs[0].need_action == False
    assert state6.uavs[1].need_action == True
    assert state6.uavs[1].cur_pose[0] == pytest.approx(8.0, 0.05) and state6.uavs[1].cur_pose[1] == pytest.approx(0.0,0.05)
    assert state6.action_cost == pytest.approx((9.153-8.472)/param.VEL_RATIO, 0.05)
    assert len(state6.get_actions()) == 1
    assert len(state6.assigned_pois) == 4
    assert state6.is_goal_state == False

    # # the sixth transition - assign action to the drone
    assert state6.get_actions()[0].target == 3
    state_prob_cost = state6.transition(state6.get_actions()[0])
    assert len(state_prob_cost) == 1
    state7 = list(state_prob_cost.keys())[0]
    assert state7.uavs[0].need_action == True
    assert state7.uavs[1].need_action == False
    assert state7.uavs[1].cur_pose[0] == pytest.approx(8.0, 0.05) and state7.uavs[1].cur_pose[1] == pytest.approx(0.0,0.05)
    assert state7.action_cost == pytest.approx((10.472-9.153)/param.VEL_RATIO, 0.05)
    assert len(state7.get_actions()) == 1
    assert len(state7.assigned_pois) == 4
    assert state7.is_goal_state == True


def test_dronestate_transition_threedrones_mdg():
    start, goal, graph = graphs.s_graph_unc()
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=param.RobotType.Drone) for _ in range(3)]
    actions = [core.Action(target=v.id, rtype=param.RobotType.Drone) for v in graph.pois if v.id != start.id]
    robot_edges = [[1,5]]
    robot_pos = [[1.0,0.0]]
    state = dstate_dec.DronesState(actions=actions, robot_pos=robot_pos, redges=robot_edges, \
                            graph=graph, restID=goal.id, drones=drones)
    assert len(state.uavs) == 3
    assert len(state.actions) == len(state.get_actions()) == 5
    assert all(uav.need_action == True for uav in state.uavs)
    assert len(state.assigned_pois) == 0

    state2 = state.copy()
    assert state2 != state
    for i in range(len(state2.uavs)):
        assert state2.uavs[i] != state.uavs[i]
    assert state2.actions == state.actions
    assert state2.assigned_pois == state.assigned_pois

    # # the first transition - assign action to the drone
    state_prob_cost = state.transition(state.get_actions()[0])
    assert state.get_actions()[0].target == 5
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    assert state1.uavs[0].need_action == False
    assert state1.uavs[1].need_action == True
    assert state1.uavs[2].need_action == True
    assert state1.uavs[0].cur_pose[0] == 0.0 and state1.uavs[0].cur_pose[1] ==0.0
    assert state1.action_cost == 0.0
    assert len(state1.get_actions()) == 4
    assert len(state1.assigned_pois) == 1
    
    # # the second transition - assign action to the drone
    assert state1.get_actions()[0].target == 6
    state_prob_cost = state1.transition(state1.get_actions()[0])
    assert len(state_prob_cost) == 1
    state2 = list(state_prob_cost.keys())[0]
    assert state2.uavs[0].need_action == False
    assert state2.uavs[1].need_action == False
    assert state2.uavs[2].need_action == True
    assert state2.uavs[0].cur_pose[0] == 0.0 and state2.uavs[0].cur_pose[1] ==0.0
    assert state2.action_cost == 0.0
    assert len(state2.get_actions()) == 3
    assert len(state2.assigned_pois) == 2
    
    # # the third transition - assign action to the drone
    assert state2.get_actions()[0].target == 7
    state_prob_cost = state2.transition(state2.get_actions()[0])
    assert len(state_prob_cost) == 1
    state3 = list(state_prob_cost.keys())[0]
    assert state3.uavs[0].need_action == False
    assert state3.uavs[1].need_action == True
    assert state3.uavs[2].need_action == False
    assert state3.uavs[1].cur_pose[0] == 2.0 and state3.uavs[1].cur_pose[1] ==0.0
    assert state3.action_cost == 2.0/param.VEL_RATIO
    assert len(state3.get_actions()) == 2
    assert len(state3.assigned_pois) == 3

    # # the forth transition - assign action to the drone
    assert state3.get_actions()[1].target == 9
    state_prob_cost = state3.transition(state3.get_actions()[1])
    assert len(state_prob_cost) == 1
    state4 = list(state_prob_cost.keys())[0]
    assert state4.uavs[0].need_action == True
    assert state4.uavs[1].need_action == False
    assert state4.uavs[2].need_action == False
    assert state4.uavs[0].cur_pose[0] == 2.0 and state4.uavs[0].cur_pose[1] ==2.0
    assert state4.action_cost ==  pytest.approx((2.824-2.0)/param.VEL_RATIO, 0.05)
    assert len(state4.get_actions()) == 1
    assert len(state4.assigned_pois) == 4
    
    # # the five transition - assign action to the drone
    assert state4.get_actions()[0].target == 8
    state_prob_cost = state4.transition(state4.get_actions()[0])
    assert len(state_prob_cost) == 1
    state5 = list(state_prob_cost.keys())[0]
    assert state5.uavs[0].need_action == False
    assert state5.uavs[1].need_action == False
    assert state5.uavs[2].need_action == True
    assert state5.uavs[2].cur_pose[0] == 4.0 and state5.uavs[2].cur_pose[1] ==2.0
    assert state5.action_cost ==  pytest.approx((4.472-2.824)/param.VEL_RATIO, 0.05)
    assert len(state5.get_actions()) == 1
    assert len(state5.assigned_pois) == 5
    assert state5.is_goal_state == False
    
    # # the sixth transition - assign action to the drone
    assert state5.get_actions()[0].target == 4
    state_prob_cost = state5.transition(state5.get_actions()[0])
    assert len(state_prob_cost) == 1
    state6 = list(state_prob_cost.keys())[0]
    assert state6.uavs[0].need_action == False
    assert state6.uavs[1].need_action == True
    assert state6.uavs[2].need_action == False
    assert state6.uavs[1].cur_pose[0] == 6.0 and state6.uavs[1].cur_pose[1] ==0.0
    assert state6.action_cost ==  pytest.approx((6.0-4.472)/param.VEL_RATIO, 0.05)
    assert len(state6.get_actions()) == 1
    assert len(state6.assigned_pois) == 5
    assert state6.is_goal_state == False
    
    # # the seventh transition - assign action to the drone
    assert state6.get_actions()[0].target == 4
    state_prob_cost = state6.transition(state6.get_actions()[0])
    assert len(state_prob_cost) == 1
    state7 = list(state_prob_cost.keys())[0]
    assert state7.uavs[0].need_action == True
    assert state7.uavs[1].need_action == False
    assert state7.uavs[2].need_action == False
    assert state7.uavs[0].cur_pose[0] == 6.0 and state7.uavs[0].cur_pose[1] ==2.0
    assert state7.action_cost ==  pytest.approx((2.824+4.0-6.0)/param.VEL_RATIO, 0.05)
    assert len(state7.get_actions()) == 1
    assert len(state7.assigned_pois) == 5
    assert state7.is_goal_state == False
    
    # # the eight transition - assign action to the drone
    assert state7.get_actions()[0].target == 4
    state_prob_cost = state7.transition(state7.get_actions()[0])
    assert len(state_prob_cost) == 1
    state8 = list(state_prob_cost.keys())[0]
    assert state8.uavs[0].need_action == False
    assert state8.uavs[1].need_action == True
    assert state8.uavs[2].need_action == False
    assert state8.uavs[1].cur_pose[0] == 8.0 and state8.uavs[1].cur_pose[1] ==0.0
    assert state8.action_cost ==  pytest.approx((8.0-2.824-4.0)/param.VEL_RATIO, 0.05)
    assert len(state8.get_actions()) == 1
    assert len(state8.assigned_pois) == 5
    assert state8.is_goal_state == False
    
    # # the nine transition - assign action to the drone
    assert state8.get_actions()[0].target == 4
    state_prob_cost = state8.transition(state8.get_actions()[0])
    assert len(state_prob_cost) == 1
    state9 = list(state_prob_cost.keys())[0]
    assert state9.uavs[0].need_action == False
    assert state9.uavs[1].need_action == False
    assert state9.uavs[2].need_action == True
    assert state9.uavs[1].cur_pose[0] == 8.0 and state9.uavs[1].cur_pose[1] ==0.0
    assert state9.action_cost ==  pytest.approx((8.944-8.0)/param.VEL_RATIO, 0.05)
    assert len(state9.get_actions()) == 1
    assert len(state9.assigned_pois) == 5
    assert state9.is_goal_state == False
    
    # # the 1oth transition - assign action to the drone
    assert state9.get_actions()[0].target == 4
    state_prob_cost = state9.transition(state9.get_actions()[0])
    assert len(state_prob_cost) == 1
    state10 = list(state_prob_cost.keys())[0]
    assert state10.uavs[0].need_action == True
    assert state10.uavs[1].need_action == False
    assert state10.uavs[2].need_action == False
    assert state10.uavs[1].cur_pose[0] == 8.0 and state10.uavs[1].cur_pose[1] ==0.0
    assert state10.action_cost ==  pytest.approx((9.6569-8.944)/param.VEL_RATIO, 0.05)
    assert len(state10.get_actions()) == 1
    assert len(state10.assigned_pois) == 5
    assert state10.is_goal_state == True