import pytest
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.robot import Robot
from sctp.param import RobotType, VEL_RATIO
# from sctp.param import EventOutcome
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
    start, goal, graph = graphs.m_graph_unc()
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    node = pomcp.POUCTNode(init_state)
    print(f"Position of robot {node.state.robot.cur_pose}")
    for i in range(1000):
        cost = core.sctp_rollout2(node)

def test_sctp_rollout3_lgraph():
    start, goal, graph = graphs.linear_graph_unc()
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    node = pomcp.POUCTNode(init_state)
    # rollout for drones
    assert core.sctp_rollout3(node) == 15.0
    
    action = core.Action(target=5, rtype=RobotType.Drone)
    state_prob_costs = init_state.transition(action)
    state = list(state_prob_costs.keys())[0]
    node = pomcp.POUCTNode(state)
    for i in range(10):
        cost = core.sctp_rollout3(node)
        assert cost == 15.0

def test_sctp_rollout3_dgraph():
    start, goal, graph = graphs.disjoint_unc()
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    node = pomcp.POUCTNode(init_state)
    # rollout for drones
    assert core.sctp_rollout3(node) == 8.0
    action = core.Action(target=5, rtype=RobotType.Drone)
    state_prob_costs = init_state.transition(action)
    state = list(state_prob_costs.keys())[0]
    node = pomcp.POUCTNode(state)
    for i in range(10):
        cost = core.sctp_rollout3(node)
        assert cost == 8.0

def test_sctp_drone_rollout_dg():
    start, goal, graph = graphs.disjoint_unc()
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    