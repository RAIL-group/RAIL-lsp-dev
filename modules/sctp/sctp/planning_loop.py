# import itertools
# import numpy as np
# import matplotlib.pyplot as plt
# import math
import copy, random   
from pouct_planner import graphs
from pouct_planner import pomdp_state
from pouct_planner import core   

def sctpbase_pomcp_navigating(nodes, edges, robots, start, goal):
   cur_pos = start
   exe_path = [start]
   total_cost = 0.0
   count = 0
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   # robot = robots #graphs.RobotData(robot_id = 1, coord_x = 1.0, coord_y = 1.0, cur_vertex=start, vel=1.0)
   initial_state = pomdp_state.SCTPBaseState(edge_probs=edge_probs, 
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   state = copy.deepcopy(initial_state)
   state, observed_status, cost = move(state) # move then sense
   update_belief_state(state, observed_status)
   while True:
      if cur_pos ==goal:
         return True, exe_path, total_cost

      action, _ = core.po_mcts(state) # apply pomcp algorithm to get the next action
      # return None, None, None
      new_obser = False
      exe_path.append(action)
      state, observed_status, cost = move(state, action) # move then sense
      total_cost += cost
      new_obser = update_belief_state(state, observed_status)
      cur_pos = state.robots.cur_vertex
      if cur_pos == goal:
         return True, exe_path, total_cost
      count += 1
      if count > 10:
         break 
   return False, exe_path, total_cost 

def move(state, action=None):
   cost = 0.0
   if action is not None:
      edge_id = tuple(sorted((state.robots.cur_vertex, action)))
      edge = [edge for edge in state.edges if edge.id == edge_id][0]
      cost = edge.get_cost()
      state.robot_move(action)
   # sense the neighbors
   neighbors = [node for node in state.vertices if node.id == state.robots.cur_vertex][0].neighbors
   observed_status = {}
   for n in neighbors:
      e_id = tuple(sorted((state.robots.cur_vertex, n)))
      next_edge = [edge for edge in state.edges if edge.id == e_id][0]
      observed_status[e_id] = float(next_edge.block_status)
   return state, observed_status, cost

def update_belief_state(state, observed_status):
   need_update = False
   for key, value in observed_status.items():
      if state.edge_probs[key] != value:
         need_update = True
      state.edge_probs[key] = value
      if state.edge_probs[key] == 1.0:
         edge = [edge for edge in state.edges if edge.id == key][0]
         edge.update_cost(10e5)
   return need_update
