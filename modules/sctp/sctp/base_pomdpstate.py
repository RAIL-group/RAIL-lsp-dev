from enum import Enum
from sctp import graphs

EventOutcome = Enum('EventOutcome', ['BLOCK', 'TRAV','CHANCE'])

class History(object):
   def __init__(self, data=None):
      self._data = data if data is not None else dict()

   def add_history(self, action, start, prob, outcome):
      assert outcome == EventOutcome.TRAV or outcome == EventOutcome.BLOCK
      self._data[(action, start, prob)] = outcome
   
   def get_action_outcome(self, action, start, prob):
      # return the history or, it it doesn't exist, return CHANCE
      return self._data.get((action, start, prob), EventOutcome.CHANCE)
   
   def copy(self):
      return History(data=self._data.copy())

   def __eq__(self, other):
      if not isinstance(other, History):
         return False
      return self._data == other._data
      
   def __str__(self):
      return f'{self._data}'
   
   def __hash__(self):
      return hash(tuple(self._data.items()))


class SCTPBaseState(object):
   def __init__(self, edge_probs, edge_costs, last_state= None, history=None, goal=None,
                     vertices=None, edges=None, robots=None):
      self.edge_probs = edge_probs
      self.edge_costs = edge_costs
      self.action_cost = 0.0
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
      # for blocking 
      block_history = self.history.copy()
      block_history.add_history(action, self.robots.cur_vertex, block_prob, EventOutcome.BLOCK)
      new_state_block = SCTPBaseState(self.edge_probs, self.edge_costs, last_state=self, history=block_history)
      new_state_block.robot_move(action)
      new_state_block.action_cost = 10e5
      belief_state[new_state_block] = (block_prob, new_state_block.action_cost)

      # for traversable
      trav_history = self.history.copy()
      trav_history.add_history(action, self.robots.cur_vertex, block_prob, EventOutcome.TRAV)
      new_state_traversable = SCTPBaseState(self.edge_probs, self.edge_costs, last_state=self, history=trav_history)
      new_state_traversable.robot_move(action)
      new_state_traversable.action_cost = self.edge_costs[edge_id]
      belief_state[new_state_traversable] = (1.0 - block_prob, new_state_traversable.action_cost)

      if new_state_block==new_state_traversable:
         raise ValueError("new_state_block==new_state_traversable")
         

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
      combined_hash = hash((graph_hash, robot_hash, self.history))
      return combined_hash

   def __hash__(self):
      self.hash_id = self.hash_state()
      return self.hash_id

   def __eq__(self, other):
      return self.hash_id == other.hash_id

   def __repr__(self):
      return f'{self.edge_probs, self.robots.cur_vertex}'

def sctpbase_rollout(state):
   # not allow the robot to move back to the parent node
   # for _ in range(n_steps):
   #    if state.is_goal_state:
   #       return state
   #    action = state.get_actions()[0]
   #    state = state.transition(action)
   pass