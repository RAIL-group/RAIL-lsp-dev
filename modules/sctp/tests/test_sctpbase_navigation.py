import pytest
from pouct_planner import core
from sctp import graphs
from sctp import base_navigation, base_pomdpstate

def test_sctpbase_nav_sense():
   # testing on a linear_graph
   start = 1
   node1 = 2
   goal = 3
   nodes = []
   node1 = graphs.Vertex(1, (0.0, 0.0))
   nodes.append(node1)
   node2 =  graphs.Vertex(2, (5.0, 0.0))
   nodes.append(node2)
   node3 =  graphs.Vertex(3, (15.0, 0.0))
   nodes.append(node3)
   
   edges = []
   edge1 =  graphs.Edge(node1, node2, 0.9)
   edge1.block_status = 1
   edges.append(edge1)
   node1.neighbors.append(node2.id)
   node2.neighbors.append(node1.id)
   edge2 =  graphs.Edge(node2, node3, 0.0)
   edge2.block_status = 0
   edges.append(edge2)
   node2.neighbors.append(node3.id)
   node3.neighbors.append(node2.id)
   robots = graphs.RobotData(robot_id = 1, position=(0.0, 0.0), cur_vertex=start)

   edge_probs = {edge.id: edge.block_prob for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, 
                     goal=goal, vertices=nodes, edges=edges, robots=robots)

   observed_status = base_navigation.sense(initial_state) # sense the env 
   assert observed_status == {(1, 2): 1.0}

def test_sctpbase_nav_update():
   pass


def test_sctpbase_nav_lineargraph():
   # testing on a linear_graph
   start = 1
   node1 = 2
   goal = 3
   nodes = []
   node1 = graphs.Vertex(1, (0.0, 0.0))
   nodes.append(node1)
   node2 =  graphs.Vertex(2, (5.0, 0.0))
   nodes.append(node2)
   node3 =  graphs.Vertex(3, (15.0, 0.0))
   nodes.append(node3)
   
   edges = []
   edge1 =  graphs.Edge(node1, node2, 0.0)
   edge1.block_status = 0
   edges.append(edge1)
   node1.neighbors.append(node2.id)
   node2.neighbors.append(node1.id)
   edge2 =  graphs.Edge(node2, node3, 0.0)
   edge2.block_status = 0
   edges.append(edge2)
   node2.neighbors.append(node3.id)
   node3.neighbors.append(node2.id)
   robots = graphs.RobotData(robot_id = 1, position=(0.0, 0.0), cur_vertex=start)

   edge_probs = {edge.id: edge.block_prob for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, 
                     goal=goal, vertices=nodes, edges=edges, robots=robots)

   state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, last_state=initial_state, history=initial_state.history)
   observed_status = base_navigation.sense(state) # sense the env
   state, _ = base_navigation.update_belief_state(state, observed_status)
   exe_path = [start]
   nav_cost = 0.0
   count = 0
   while True:
      if state.is_goal_state:
         return True, exe_path, nav_cost

      action, exp_cost = core.po_mcts(state) # apply pomcp algorithm to get the next action
      exe_path.append(action)
      state, move_cost = base_navigation.move(state, action) # move then sense
      nav_cost += move_cost
      observed_status = base_navigation.sense(state)

      state, new_obser = base_navigation.update_belief_state(state, observed_status)
      count += 1
      if count > 10:
         break 
   return False, exe_path, nav_cost


   # print(f"the action is {action}")
   # edge_id = tuple(sorted((start, action)))
   # print(f"the prob is {initial_state.edge_probs[edge_id]}")
   # print("-----------------------------")
   # initial_state.robot_move(action)
   # actions = initial_state.get_actions()
   # print(f"the action is {actions}")
   # for act in actions:
   #    print(f"the action is {act}")
   #    edge_id = tuple(sorted((action, act)))
   #    print(f"the prob is {initial_state.edge_probs[edge_id]}")