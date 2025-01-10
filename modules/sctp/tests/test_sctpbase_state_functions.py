import pytest
from pouct_planner import core
from sctp import graphs, base_pomdpstate
from sctp.base_pomdpstate import EventOutcome

def test_sctpbase_state_transition():
   start = 1
   node1 = 2
   goal = 3
   prob = 0.0
   nodes = []
   node1 = graphs.Vertex(1, (0.0, 0.0))
   nodes.append(node1)
   node2 =  graphs.Vertex(2, (5.0, 0.0))
   nodes.append(node2)
   node3 =  graphs.Vertex(3, (15.0, 0.0))
   nodes.append(node3)
   
   edges = []
   edge1 =  graphs.Edge(node1, node2, prob)
   edge1.block_status = 0
   edges.append(edge1)
   node1.neighbors.append(node2.id)
   node2.neighbors.append(node1.id)
   edge2 =  graphs.Edge(node2, node3, prob)
   edge2.block_status = 0
   edges.append(edge2)
   node2.neighbors.append(node3.id)
   node3.neighbors.append(node2.id)
   robots = graphs.RobotData(robot_id = 1, position=(0.0, 0.0), cur_vertex=start)

   edge_probs = {edge.id: edge.block_prob for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, 
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 1
   action = all_actions[0]
   assert action == 2

   outcome_states = initial_state.transition(action)
   assert len(outcome_states) == 2
   for state, (prob, cost) in outcome_states.items():
      if prob == 0.0:
         assert state.history.get_action_outcome(action, start, prob) == EventOutcome.BLOCK
         assert cost == pytest.approx(10e5, abs=0.1)
      if prob ==1.0:
         assert state.history.get_action_outcome(action, start, 1.0-prob) == EventOutcome.TRAV
         assert cost == pytest.approx(5.0, abs=0.1)   

def test_sctpbase_state_cost():
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
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 1
   action = all_actions[0]
   assert action == 2

   best_action, cost = core.po_mcts(initial_state, n_iterations=2000)
   assert best_action == 2
   assert cost == pytest.approx(15.0, abs=0.1)


def test_sctpbase_state_transition_probs():
   start = 1
   goal = 3
   # this disjoint graph have 4 nodes (1,2,3,4) and 4 edges: (1,4), (1,2), (3,4), (2,3)
   nodes = []
   node1 =  graphs.Vertex(1, (0.0, 0))
   nodes.append(node1)
   node2 =  graphs.Vertex(2, (0.0, 4.0))
   nodes.append(node2)
   node3 =  graphs.Vertex(3, (4.0, 4.0))
   nodes.append(node3)
   node4 =  graphs.Vertex(4, (5.0, 4.0))
   nodes.append(node4)
   
   edges = []
   # edge 1
   edge1 =  graphs.Edge(node1, node2, 0.1)
   edge1.block_status = 0
   edges.append(edge1)
   node1.neighbors.append(node2.id)
   node2.neighbors.append(node1.id)
   # edge 2
   edge2 =  graphs.Edge(node3, node4, 0.9)
   edge2.block_status = 1
   edges.append(edge2)
   node3.neighbors.append(node4.id)
   node4.neighbors.append(node3.id)
   # edge 3
   edge3 =  graphs.Edge(node1, node4, 0.1) # length = 6.4
   edge3.block_status = 0
   edges.append(edge3)
   node1.neighbors.append(node4.id)
   node4.neighbors.append(node1.id)
   # edge 4
   edge4 =  graphs.Edge(node2, node3, 0.1)
   edge4.block_status = 0
   edges.append(edge4)
   node2.neighbors.append(node3.id)
   node3.neighbors.append(node2.id)
   robots = graphs.RobotData(robot_id = 1, position=(0.0, 0.0), cur_vertex=start)
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, 
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 2
   action = all_actions[0]
   assert action == 2 or action == 4

   outcome_states = initial_state.transition(action)
   assert len(outcome_states) == 2
   for state, (prob, cost) in outcome_states.items():
      if action == 2:
         if state.history.get_action_outcome(action, start, prob) == EventOutcome.BLOCK:
            assert prob == pytest.approx(0.1)
            assert cost == pytest.approx(10e5, abs=0.1)
         else:
            assert prob == pytest.approx(0.9)
            assert cost == pytest.approx(4.0, abs=0.1)
      elif action == 4:
         if state.history.get_action_outcome(action, start, prob) == EventOutcome.BLOCK:
            assert cost == pytest.approx(10e5, abs=0.1)
         else:
            assert cost == pytest.approx(6.4, abs=0.1)