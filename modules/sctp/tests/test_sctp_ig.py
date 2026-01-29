import pytest, random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sctp import sctp_graphs as graphs
import sctp.action_estimation as ae 
from sctp.utils import plotting
from sctp import param
from sctp import core

def test_sctp_ig_djgraph():
   start, goal, graph = graphs.disjoint_unc()
   poi_check = graph.pois[0]
   poi6 = graph.pois[1]
   poi7 = graph.pois[2]
   poi8 = graph.pois[3]
   poi_check.block_prob = 0.0 
   poi6.block_prob = 0.4
   poi8.block_prob = 0.0
   poi7.block_prob = 0.5
   V2 = graph.vertices[1]
   V3 = graph.vertices[2]
   V4 = graph.vertices[3]
   # print(f"The node to check with ID: {poi_check.id}")
   # robot_edge = [start.id, start.id]
   robot_edge = [start.id, poi_check.id]
   d0 = 1.0
   d1 = 1.0
   num_samples = 15000
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   assert heuristic == pytest.approx(47.196, abs=1.0)
   # print(f"Heuristic: {heuristic}")
   action = core.Action(target = poi_check.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   assert cost == pytest.approx(-d0, abs=0.5)


def test_sctp_ig_sgraph():
   print("")
   start, goal, graph = graphs.s_graph_unc()
   poi5 = graph.pois[0]
   poi6 = graph.pois[1]
   poi7 = graph.pois[2]
   poi8 = graph.pois[3]
   poi9 = graph.pois[4]   
   V2 = graph.vertices[1]
   V3 = graph.vertices[2]
   V4 = graph.vertices[3]
   action = core.Action(target = poi7.id)
   num_samples = 15000
   # if all edges are passable, edge 7's not not positive
   poi6.block_prob = 0.0
   poi5.block_prob = 0.0
   poi7.block_prob = 0.5
   poi8.block_prob = 0.0
   poi9.block_prob = 0.0
   robot_edge = [start.id, start.id]
   d0 = 0.0
   d1 = 0.0
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   assert cost == pytest.approx(-2.24, abs=0.5)
   print(f"Cost: {cost}")
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   
   poi6.block_prob = 1.0
   poi5.block_prob = 0.0
   poi7.block_prob = 0.5
   poi8.block_prob = 0.0
   poi9.block_prob = 1.0
   robot_edge = [start.id, start.id]
   d0 = 0.0
   d1 = 0.0
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   assert cost == pytest.approx(-2.24, abs=0.5)
   print(f"Cost: {cost}")
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   
   # if all ways to goal are blocked, edge 7's cost is not positive, too
   poi8.block_prob = 1.0
   poi9.block_prob = 1.0
   poi6.block_prob = 0.2
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   # print(f"Heuristic: {heuristic}")
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)   
   print(f"Cost: {cost}")
   assert cost == pytest.approx(-2.24, abs=0.5)
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   # if 6,9 are blocked, edge 7's cost is not positive, too
   poi8.block_prob = 0.1
   poi9.block_prob = 1.0
   poi6.block_prob = 1.0
   poi5.block_prob = 0.4
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)   
   print(f"Cost: {cost}")
   assert cost == pytest.approx(-2.24, abs=1.0) or cost < 0.0
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   # if 6,8 have highly blocked likelihood, edge 7's has most value
   poi8.block_prob = 0.5
   poi9.block_prob = 0.9
   poi6.block_prob = 0.5
   poi5.block_prob = 0.9

   poi7.block_prob = 0.1
   num_samples = 20000
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   cost3 = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)   
   # print(f"Cost3: {cost3}")
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   poi7.block_prob = 0.5
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   # print(f"Heuristic: {heuristic}")
   cost1 = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)   
   # print(f"Cost1: {cost1}")
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()

   poi7.block_prob = 0.8
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   cost2 = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)   
   # print(f"Cost2: {cost2}")
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   
   assert cost1 > cost3
   assert cost1 > cost2


def test_sctp_middle_sgraph():
   print("")
   start, goal, graph = graphs.s_graph_unc()
   poi5 = graph.pois[0]
   poi6 = graph.pois[1]
   poi7 = graph.pois[2]
   poi8 = graph.pois[3]
   poi9 = graph.pois[4]   
   V2 = graph.vertices[1]
   V3 = graph.vertices[2]
   V4 = graph.vertices[3]
   action = core.Action(target = poi7.id)
   num_samples = 50000
   # if all edges are passable, edge 7's not not positive
   poi6.block_prob = 0.1
   poi5.block_prob = 0.1
   poi7.block_prob = 0.6
   poi8.block_prob = 0.1
   poi9.block_prob = 0.1
   robot_edge = [poi6.id, V3.id]
   d0 = 1.0
   d1 = 1.0
   atNode = False
   drone_pose = (3.0, 0.0)
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=atNode, startNode=start.id,
                                     n_samples=num_samples)
   b1 = 0.9*(0.1*200.0+ 0.9*(0.1*200+0.9*(8*1.41421356+ 3)))
   h = (1.0-poi9.block_prob)*5 + poi9.block_prob*((1.0-poi7.block_prob)*(0.9*(4*1.41421356+ 5)+0.1*200) + poi7.block_prob*(0.1*200+b1))
   assert heuristic == pytest.approx(h, abs=0.5)
   pass_a7 = 0.9*5 + 0.1*(0.9*(4*1.41421356+ 5)+0.1*200)
   block_a7 = 0.9*5 + 0.1*(b1+0.1*200)
   bh, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   cost_true  = h - pass_a7*(1.0-poi7.block_prob) - block_a7*poi7.block_prob - 2.236/param.VEL_RATIO
   assert cost == pytest.approx(cost_true, abs=0.5)
   

   
def test_sctp_ig_island_sgraph():
   print("")
   start, goal, graph = graphs.island_sgraph()
   num_samples = 15000
   poi9 = graph.pois[0]
   poi10 = graph.pois[1]
   poi11 = graph.pois[2]
   poi12 = graph.pois[3]
   poi13 = graph.pois[4]
   poi14 = graph.pois[5]
   poi15 = graph.pois[6]
   poi18 = graph.pois[9]
   poi16 = graph.pois[7]
   poi17 = graph.pois[8]
   poi19 = graph.pois[10]   
   # if all edges are passable, edge 7's not not positive
   poi9.block_prob = 0.0
   robot_edge = [start.id, start.id]
   d0 = 0.0
   d1 = 0.0
   
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=True, startNode=start.id,
                                     n_samples=num_samples)
   print(f"Heuristic: {heuristic}")
   action = core.Action(target = poi9.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   # assert cost == pytest.approx(-2.24, abs=0.5)
   print(f"Action {action.target} has a cost of : {cost}")
   
   action = core.Action(target = poi14.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")
   
   action = core.Action(target = poi15.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")
   
   action = core.Action(target = poi10.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")

   action = core.Action(target = poi11.id)
   
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()

def test_sctp_ig_island_bridges_sgraph():
   print("")
   starts, goals, graph = graphs.island_bridges_sgraph()
   num_samples = 200
   poi8 = graph.pois[0]
   poi9 = graph.pois[1]
   poi10 = graph.pois[2]
   poi11 = graph.pois[3]
   poi12 = graph.pois[4]
   poi13 = graph.pois[5]
   poi14 = graph.pois[6]
   poi15 = graph.pois[7]
   poi16 = graph.pois[8]   
   poi17 = graph.pois[9]
   poi9.block_prob = 0.0
   # if drone is moved
   robot_edge = [poi9.id, starts[0].id]
   d0 = 5.0
   d1 = 5.0
   atNode = False
   # cur_robot_node = starts[0].id
   # cur_drone_pose = (2.0,3.0)
   # cur_robot_pose = (0.0,0.0)
   start_node = poi9
   
   heuristic = 200.0
   # print(f"Heuristic: {heuristic}")
   action = core.Action(target = poi8.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi8.block_prob
   dist = np.linalg.norm(np.array(start_node.coord)-np.array(poi8.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*(param.VEL_RATIO/dist)
   print(f"Action {action.target} has behave changes : {joint_bc:.2f}")
      
   # action = core.Action(target = poi9.id)
   # behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"Action {action.target} has behave changes : {behave_change:.2f}")
   
   # block_prob = poi9.block_prob
   # dist = np.linalg.norm(np.array(start_node.coord)-np.array(poi9.coord))
   # joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   # print(f"Action {action.target} has behave changes : {joint_bc:.2f}")
   
   action = core.Action(target = poi10.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi10.block_prob
   dist = np.linalg.norm(np.array(start_node.coord)-np.array(poi10.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes : {joint_bc:.2f}")
   
   action = core.Action(target = poi11.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi11.block_prob
   dist = np.linalg.norm(np.array(start_node.coord)-np.array(poi11.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes : {joint_bc:.2f}")
   
   action = core.Action(target = poi12.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi12.block_prob
   dist = np.linalg.norm(np.array(start_node.coord)-np.array(poi12.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes : {joint_bc:.2f}")
   
   action = core.Action(target = poi13.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi13.block_prob
   dist = np.linalg.norm(np.array(start_node.coord)-np.array(poi13.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes : {joint_bc:.2f}")
   
   action = core.Action(target = poi14.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi14.block_prob
   dist = np.linalg.norm(np.array(start_node.coord)-np.array(poi14.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes : {joint_bc:.2f}")
   
   
      
   fig, ax = plt.subplots()
   ax.set_aspect('equal', adjustable='box')
   count =1
   for start, goal in zip(starts, goals):
        plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
        plt.text(start.coord[0]-1.8, start.coord[1], f'S{count}', fontsize=8)
        plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
        plt.text(goal.coord[0]+1.4, goal.coord[1], f'G{count}', fontsize=8)
        count += 1
   plotting.plot_sctpgraph(graph=graph, plt=ax, verbose=True)
   plt.show()
    
def test_sctp_ig_island_bridges_graph():
   print("")
   starts, goals, graph = graphs.get_insland_bridges_graph()
   num_samples = 200
   poi17 = graph.pois[0]
   poi18 = graph.pois[1]
   poi19 = graph.pois[2]
   poi20 = graph.pois[3]
   poi21 = graph.pois[4]
   poi22 = graph.pois[5]
   poi23 = graph.pois[6]
   poi24 = graph.pois[7]
   poi25 = graph.pois[8]   
   poi26 = graph.pois[9]
   poi27 = graph.pois[10]
   poi28 = graph.pois[11]
   poi29 = graph.pois[12]
   poi30 = graph.pois[13]
   poi31 = graph.pois[14]
   poi32 = graph.pois[15]
   poi33 = graph.pois[16]
   poi34 = graph.pois[17]
   poi35 = graph.pois[18]
   poi36 = graph.pois[19]
   poi37 = graph.pois[20]
   poi38 = graph.pois[21]
   poi39 = graph.pois[22]
   poi40 = graph.pois[23]
   poi41 = graph.pois[24]
   poi42 = graph.pois[25]
   poi43 = graph.pois[26]
   poi44 = graph.pois[27]
   poi45 = graph.pois[28]
   poi46 = graph.pois[29]
   poi47 = graph.pois[30]
   poi23.block_prob = 1.0
   # if drone is moved
   robot_edge = [starts[0].id, poi23.id]
   d0 = 8.0
   d1 = 8.0
   atNode = False
   # cur_robot_node = starts[0].id
   # cur_drone_pose = (2.0,3.0)
   # cur_robot_pose = (0.0,0.0)
   # start_node = starts[0]
   drone_pose = poi23
   
   heuristic = 400.0
   # print(f"Heuristic: {heuristic}")
   action = core.Action(target = poi17.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi17.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi17.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*(param.VEL_RATIO/dist)
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value: {joint_bc:.2f}")
      
   action = core.Action(target = poi18.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi18.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi18.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*(param.VEL_RATIO/dist)
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value : {joint_bc:.2f}")
   
   action = core.Action(target = poi19.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi19.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi19.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value: {joint_bc:.2f}")
   
   action = core.Action(target = poi20.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi20.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi20.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value: {joint_bc:.2f}")
   
   action = core.Action(target = poi21.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi21.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi21.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value: {joint_bc:.2f}")
   
   action = core.Action(target = poi22.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi22.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi22.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value: {joint_bc:.2f}")
   
   # action = core.Action(target = poi23.id)
   # behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   # block_prob = poi23.block_prob
   # dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi23.coord))
   # joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   # print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value : {joint_bc:.2f}")
   
   action = core.Action(target = poi24.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi24.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi24.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value : {joint_bc:.2f}")
   
   action = core.Action(target = poi25.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi25.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi25.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value : {joint_bc:.2f}")

   action = core.Action(target = poi26.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi26.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi26.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value : {joint_bc:.2f}")

   action = core.Action(target = poi27.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi27.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi27.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value : {joint_bc:.2f}")

   action = core.Action(target = poi28.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi28.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi28.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value : {joint_bc:.2f}")

   action = core.Action(target = poi29.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi29.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi29.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value : {joint_bc:.2f}")
   
   action = core.Action(target = poi30.id)
   behave_change = ae.get_single_behavior_change(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goals[0].id, cur_heuristic=heuristic, n_samples=num_samples)
   block_prob = poi30.block_prob
   dist = np.linalg.norm(np.array(drone_pose.coord)-np.array(poi30.coord))
   joint_bc = behave_change*(1.0 - block_prob)* block_prob*param.VEL_RATIO/dist
   print(f"Action {action.target} has behave changes: {behave_change:.2f} joint value: {joint_bc:.2f}")
   
   
      
   fig, ax = plt.subplots()
   ax.set_aspect('equal', adjustable='box')
   count =1
   for start, goal in zip(starts, goals):
        plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
        plt.text(start.coord[0]-1.8, start.coord[1], f'S{count}', fontsize=8)
        plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
        plt.text(goal.coord[0]+1.4, goal.coord[1], f'G{count}', fontsize=8)
        count += 1
   plotting.plot_sctpgraph(graph=graph, plt=ax, verbose=False, initG=True)
   plt.show()
    