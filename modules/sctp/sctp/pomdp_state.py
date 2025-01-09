# import itertools
# import numpy as np
from pouct_planner import graphs

class SCTPBaseState(object):
   def __init__(self, edge_probs, goal=None, start_state= None,
                     vertices=None, edges=None, robots=None):
      # print("SCTPBaseState")
      self.edge_probs = edge_probs
      self.action_cost = 0.0
      self.block_status = False
      if start_state is None:
         self.vertices = vertices
         self.edges = edges
         self.goalID = goal 
         self.robots = robots
      else:
         self.vertices = start_state.vertices
         self.edges = start_state.edges
         self.goalID = start_state.goalID
         self.robots = graphs.RobotData(last_robot=start_state.robots)
      self.actions = [node for node in self.vertices if node.id == self.robots.cur_vertex][0].neighbors


   def get_actions(self):
      return self.actions
   @property
   def is_terminal(self):
      return self.robots.cur_vertex == self.goalID


   def transition(self, action):
      return self.robot_transition(action) 
   

   def robot_transition(self, action):
      belief_state = {}
      edge_id = tuple(sorted((self.robots.cur_vertex, action)))
      self.robots.last_vertex = self.robots.cur_vertex
      block_prob = self.edge_probs[edge_id]

      self.robots.cur_vertex = action
      new_state_block = SCTPBaseState(self.edge_probs, start_state=self)
      new_state_block.action_cost = 10e5
      # new_state_block.edge_probs[edge_id] = 1.0
      new_state_block.block_status = True
      belief_state[new_state_block] = (block_prob, new_state_block.action_cost)

      new_state_traversable = SCTPBaseState(self.edge_probs, start_state=self)
      new_state_traversable.action_cost = [edge for edge in self.edges if edge.id == edge_id][0].cost
      new_state_traversable.block_status = False
      # new_state_traversable.edge_probs[edge_id] = 0.0
      belief_state[new_state_traversable] = (1 - block_prob, new_state_traversable.action_cost)

      return belief_state
      
