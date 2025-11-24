import random
import numpy as np
import argparse
from sctp import sctp_graphs as graphs
from sctp import core
# from pouct_planner import core as policy
from sctp.robot import Robot
from sctp.param import EventOutcome, RobotType
from sctp.planners import sctp_planner
from sctp import dstate_dec
from sctp.planners.sctp_planner_dec import DronesPlanner, GroundPlanner

def test_sctp_planner_lg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='base')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--C', type=int, default=100)
    parser.add_argument('--resolution', type=float, default=0.05)
    
    args = parser.parse_args()
    args.current_seed = args.seed
    # print(args.num_iterations)
    exp_param=50.0
    start, goal, l_graph = graphs.linear_graph_unc()
    for poi in l_graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
    robot = Robot(position=[0.0, 0.0], cur_node=start.id)
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=RobotType.Drone)]
    sctpplanner = sctp_planner.SCTPPlanner(args=args, init_graph=l_graph, 
                                    goalID=goal.id, robot=robot, drones=drones) 
    actions, costs = sctpplanner.compute_joint_action()
    vertices_status = {4: 0}
    
    robot_data = ([[15.0, 0], True, None, 3])
    drone_data = ([[[15.0,0], True, 3]])
    test_data = {"observed_pois": (vertices_status), "robot": robot_data, "drones": drone_data}
    sctpplanner.update(test_data['observed_pois'], test_data['robot'], test_data['drones'])
    if sctpplanner.reached_goal():
        print("The robot reaches it goals")

def test_sctp_planner_sg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='base')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--C', type=int, default=100)
    parser.add_argument('--resolution', type=float, default=0.05)
    
    args = parser.parse_args()
    args.current_seed = args.seed
    # print(args.num_iterations)
    exp_param=50.0
    start, goal, s_graph = graphs.s_graph_unc()
    for poi in s_graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
    robot = Robot(position=[0.0, 0.0], cur_node=start.id)
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=RobotType.Drone)]
    sctpplanner = sctp_planner.SCTPPlanner(args=args, init_graph=s_graph, 
                                    goalID=goal.id, robot=robot, drones=drones) 
    actions, costs = sctpplanner.compute_joint_action()
    for action in actions:
        print(action)
    vertices_status = {5: 0}
    
    robot_data = ([[8.0, 0], True, None, 4])
    drone_data = ([[[8.0,0], True, 4]])
    test_data = {"observed_pois": (vertices_status), "robot": robot_data, "drones": drone_data}
    sctpplanner.update(test_data['observed_pois'], test_data['robot'], test_data['drones'])
    if sctpplanner.reached_goal():
        print("The robot reaches it goals")


def test_dronesplanner_sg(args):    
    start, goal, s_graph = graphs.s_graph_unc()
    tree_depth = 20
    verbose = False
    num_drones = args.num_drones
    num_drones = 2
    args.num_iterations = 2000
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=RobotType.Drone) for _ in range(num_drones)]
    for i, d in enumerate(drones):
        d.need_action = True
        if i==0:
            d.unfinished_action = core.Action(target=8, rtype=RobotType.Drone, start_pose=(0.0,0.0))
        else:
            # d.unfinished_action = None
            d.unfinished_action = core.Action(target=9, rtype=RobotType.Drone, start_pose=(0.0,0.0))
    actions = [core.Action(target=v.id, rtype=RobotType.Drone,  start_pose = (0.0,0.0)) for v in s_graph.pois if v.id != start.id]
    drone_planner = DronesPlanner(args=args, init_graph=s_graph, goalID=goal.id, drones=drones,
                                  rollout_fn=dstate_dec.drone_rollout, verbose=verbose,tree_depth=tree_depth)
    drone_planner.robots_edges = [[1,5]]
    drone_planner.robots_pos = [[1.0,1.0]]
     
    actions, costs = drone_planner.compute_joint_action(sub_actions=actions)
    for action in actions:
        print(action)
    
    # update function
    vertices_status = {5: 0}
    robot_data = ([[[0.0, 0.0], False, [1,1], 1,1]])
    drone_data = ([[[8.0,0.0], True, 4, None],[[6.0,2.0], True, 7, None]])
    test_data = {"observed_pois": (vertices_status), "robots": robot_data, "drones": drone_data}
    print(f"The robot data is: {robot_data}")
    drone_planner.update(observations=test_data['observed_pois'], drone_data=test_data['drones'], robot_data=test_data['robots'])
    

def test_groundplanner_sg(args):    
    start, goal, s_graph = graphs.s_graph_unc()
    tree_depth = 20
    verbose = False
    args.num_iterations = 1000
    # robot.at_node = True
    robot = Robot(position=[1.0, 1.0], cur_node=start.id, at_node=False, edge=[1,5], robot_type=RobotType.Ground)
    robot_planner = GroundPlanner(args=args, init_graph=s_graph, goalID=goal.id, robot=robot,
                                  rollout_fn=core.sctp_rollout3, tree_depth=tree_depth,verbose=verbose)
    
    actions, costs = robot_planner.compute_action()
    for action in actions:
        print(action)
    
    # update function
    vertices_status = {5: 0}
    robot_data = ([[[0.0, 0.0], False, [1,1], 1,1]])
    drone_data = ([[[8.0,0], True, 4, None],[[6.0,2.0], True, 7, None]])
    test_data = {"observed_pois": (vertices_status), "robot": robot_data, "drones": drone_data}
    robot_planner.update(observations=test_data['observed_pois'], robot_data=test_data['robot'][0])




if __name__ == '__main__':
#     # test_sctp_planner_lg()
#     test_sctp_planner_sg()    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='base')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=20)
    parser.add_argument('--C', type=float, default=200)
    args = parser.parse_args()
    args.current_seed = args.seed
    
    
    # test_dronesplanner_sg(args=args)
    test_groundplanner_sg(args=args)

