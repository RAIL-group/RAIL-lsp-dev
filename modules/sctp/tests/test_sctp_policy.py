import random
import numpy as np
from sctp import sctp_graphs as graphs
from sctp import core
from pouct_planner import core as policy
from sctp.robot import Robot
from sctp.param import EventOutcome, RobotType

def test_sctp_policy_lg():
    exp_param=50.0
    start, goal, l_graph = graphs.linear_graph_unc()
    robot = Robot(position=[0.0, 0.0], cur_node=start.id)
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=RobotType.Drone)]
    init_state = core.SCTPState(graph=l_graph, goalID=goal.id, robot=robot, drones=drones)
    # ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=10000)
    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=100,\
                                                rollout_fn=core.sctp_rollout)
    for p in pc[0]:
        print(p)
    for c in pc[1]:
        print(c)

def test_sctp_policy_dg():
    seed = random.randint(10,999)
    np.random.seed(seed)
    exp_param=50.0
    start, goal, d_graph = graphs.disjoint_unc()
    robot = Robot(position=[0.0, 0.0], cur_node=start.id)
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=RobotType.Drone)]
    init_state = core.SCTPState(graph=d_graph, goal=goal.id, robot=robot, drones=drones)
    assert init_state.robot.need_action == True 
    assert init_state.uavs[0].need_action == True
    # ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=1000)
    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=1000,\
                                             rollout_fn=core.sctp_rollout2)
    reach_goal = False
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    if reach_goal:
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
        print("The ground robot reaches its goal")
    graphs.plot_sctpgraph(d_graph.vertices, d_graph.pois, d_graph.edges, path=pc[0], 
                                    startID=start.id, goalID=goal.id, seed=seed)
    # for c in pc[1]:
    #     print(c)
    

def test_sctp_policy_sg():
    seed = random.randint(10,999)
    np.random.seed(seed)
    exp_param=50.0
    start, goal, s_graph = graphs.s_graph_unc()
    robot = Robot(position=[0.0, 0.0], cur_node=start.id)
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=RobotType.Drone)]
    init_state = core.SCTPState(graph=s_graph, goalID=goal.id, robot=robot, drones=drones)
    # ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=1000)
    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=2000,\
                                             rollout_fn=core.sctp_rollout2)
    reach_goal = False
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    if reach_goal:
        print("The ground robot reaches its goal")
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
    graphs.plot_sctpgraph(s_graph.vertices, s_graph.pois, s_graph.edges, path=pc[0], 
                                    startID=start.id, goalID=goal.id, seed=seed)    

def test_sctp_policy_mg():
    seed = random.randint(10,999)
    # seed = 665
    np.random.seed(seed)
    exp_param=50.0
    start, goal, m_graph = graphs.m_graph_unc()
    robot = Robot(position=[start.coord[0],start.coord[1]], cur_node=start.id)
    drones = [Robot(position=[start.coord[0],start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone)]
    init_state = core.SCTPState(graph=m_graph, goalID=goal.id, robot=robot, drones=drones)
    # ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=5000)
    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=5000,\
                                             rollout_fn=core.sctp_rollout2)
    reach_goal = False
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    if reach_goal:
        print("The ground robot reaches its goal")
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
    graphs.plot_sctpgraph(m_graph.vertices, m_graph.pois, m_graph.edges, path=pc[0], 
                                    startID=start.id, goalID=goal.id, seed=seed)    