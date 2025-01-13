import pytest
from pouct_planner import core
from sctp import graphs, base_pomdpstate, base_navigation
from sctp.base_pomdpstate import EventOutcome

def test_sctpbase_state_cost():
   start, goal, nodes, edges, robots = graphs.linear_graph_unc()
   
   block_probs = [0.0, 0.5, 0.9, 1.0]
   for prob in block_probs:
      edge12 = [edge for edge in edges if edge.id == (1, 2)][0]
      edge12.block_prob = prob
      edge_probs = {edge.id: edge.block_prob for edge in edges}
      edge_costs = {edge.id: edge.cost for edge in edges}
      initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs,
                        goal=goal, vertices=nodes, edges=edges, robots=robots)
      all_actions = initial_state.get_actions()
      assert len(all_actions) == 1
      action = all_actions[0]
      assert action == 2

      belief_state = initial_state.transition(action)
      assert len(belief_state) == 2
      expected_cost = 0.0
      for _, (p, cost) in belief_state.items():
         expected_cost += p * cost
      prob = edge_probs[(1, 2)]
      assert expected_cost == pytest.approx (10e5*prob + 5.0*(1.0-prob), abs=0.1)
   

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
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs, 
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

### This test ensures there are two states created after each action
def test_sctpbase_state_functions_sgraph(): 
   # testing on a regular graph
   start, goal, nodes, edges, robots = graphs.s_graph_unc() # edge 34 is blocked.

   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   robots.cur_vertex = 3
   state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs, 
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   assert state.robots.cur_vertex == 3
   assert len(state.get_actions()) == 4
   observed_status = base_navigation.sense(state) # sense the env
   assert observed_status == {(2, 3): 0.0, (3, 4): 1.0, (3, 5): 0.0, (1,3): 0.0}
   state, _ = base_navigation.update_belief_state(state, observed_status)
   belief_state = state.transition(4)
   assert len(belief_state) == 2