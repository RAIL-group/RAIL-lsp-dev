# import itertools
# import numpy as np
from enum import Enum
from sctp import graphs

EventOutcome = Enum('EventOutcome', ['BLOCK', 'CHANCE', 'TRAV'])

class History(object):
   def __init__(self, data=None):
      self._data = data if data is not None else dict()

   def add_history(self, action, outcome):
      assert outcome == EventOutcome.TRAV or outcome == EventOutcome.BLOCK
      self._data[action] = outcome
   
   def get_action_outcome(self, action):
      # return the history or, it it doesn't exist, return CHANCE
      raiseError ValueError("Action not in history")
      return self._data.get(action, None) #??????????????????????/
   
   def copy(self):
      return History(data=self._data.copy())

   def __eq__(self, other):
      if not isinstance(other, History):
         return False
      return self._data == other._data
      
   def __str__(self):
      return f'{self._data}'


class SCTPBaseState(object):
   def __init__(self, edge_probs, last_state= None, history=None, goal=None,
                     vertices=None, edges=None, robots=None):
      # print("SCTPBaseState")
      self.edge_probs = edge_probs
      self.action_cost = 0.0
      # self.block_status = False
      if history is None:
         self.history = History()
      else:
         self.history = history
      if last_state is None:
         self.vertices = vertices
         self.edges = edges
         self.goalID = goal 
         self.robots = robots
      else:
         self.vertices = last_state.vertices
         self.edges = last_state.edges
         self.goalID = last_state.goalID
         self.robots = graphs.RobotData(last_robot=last_state.robots)
      self.actions = [node for node in self.vertices if node.id == self.robots.cur_vertex][0].neighbors


   def get_actions(self):
      return self.actions
   
   @property
   def is_goal_state(self):
      return self.robots.cur_vertex == self.goalID


   def transition(self, action):
      return self.robot_transition(action) 
   

   def robot_transition(self, action):
      belief_state = {}
      edge_id = tuple(sorted((self.robots.cur_vertex, action)))
      block_prob = self.edge_probs[edge_id]

      new_state_block = SCTPBaseState(self.edge_probs, start_state=self)
      new_state_block.robot_move(action)
      new_state_block.action_cost = 10e5
      new_state_block.block_status = True
      
      belief_state[new_state_block] = (block_prob, new_state_block.action_cost)

      new_state_traversable = SCTPBaseState(self.edge_probs, start_state=self)
      new_state_traversable.robot_move(action)
      new_state_traversable.action_cost = [edge for edge in self.edges if edge.id == edge_id][0].cost
      new_state_traversable.block_status = False
      
      belief_state[new_state_traversable] = (1.0 - block_prob, new_state_traversable.action_cost)


      return belief_state
      
   def robot_move(self, action):
      self.robots.last_vertex = self.robots.cur_vertex
      self.robots.cur_vertex = action
      self.actions = [vertex for vertex in self.vertices if vertex.id == self.robots.cur_vertex][0].neighbors

   def hash_graph(self):
      edges_tuple = tuple(self.edge_probs)
      # Compute the hash
      return hash(edges_tuple)

   def hash_robot(self):
      return hash(self.robots.cur_vertex)
   
   def hash_state(self):
      graph_hash = self.hash_graph()
      robot_hash = self.hash_robot()
      # Combine the two hashes
      combined_hash = hash((graph_hash, robot_hash, self.action_cost))
      return combined_hash

   def __hash__(self):
      self.hash_id = self.hash_state()
      return self.hash_id

   def __eq__(self, other):
      return self.hash_id == other.hash_id

   def __repr__(self):
      return f'{self.edge_probs, self.robots.cur_vertex}'

      