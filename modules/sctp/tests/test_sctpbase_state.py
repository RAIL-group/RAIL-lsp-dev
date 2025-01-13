import pytest
from sctp import base_pomdpstate, base_navigation
from pouct_planner import core
from sctp import graphs

def test_sctpbase_state_lineargraph():
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
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs,
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 1
   action = all_actions[0]
   assert action == 2

   best_action, cost = core.po_mcts(initial_state, n_iterations=2000)
   assert best_action == 2
   assert cost == pytest.approx(15.0, abs=0.1)


def test_sctpbase_state_disjointgraph_noblock():
   start = 1
   goal = 3
   # this disjoint graph have 4 nodes (1,2,3,4) and 4 edges: (1,4), (1,2), (3,4), (2,3)
   nodes = []
   node1 =  graphs.Vertex(1, (0.0, 0.0))
   nodes.append(node1)
   node2 =  graphs.Vertex(2, (0.0, 4.0))
   nodes.append(node2)
   node3 =  graphs.Vertex(3, (4.0, 4.0))
   nodes.append(node3)
   node4 =  graphs.Vertex(4, (5.0, 4.0))
   nodes.append(node4)
   
   edges = []
   edge1 =  graphs.Edge(node1, node2, 0.0)
   edge1.block_status = 0
   edges.append(edge1)
   node1.neighbors.append(node2.id)
   node2.neighbors.append(node1.id)
   edge2 =  graphs.Edge(node3, node4, 0.0)
   edge2.block_status = 0
   edges.append(edge2)
   node3.neighbors.append(node4.id)
   node4.neighbors.append(node3.id)
   edge7 =  graphs.Edge(node1, node4, 0.0)
   edge7.block_status = 1
   edges.append(edge7)
   node1.neighbors.append(node4.id)
   node4.neighbors.append(node1.id)
   edge8 =  graphs.Edge(node2, node3, 0.0)
   edge8.block_status = 0
   edges.append(edge8)
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
      if prob == 0.0:
         assert state.history.get_action_outcome(action, start, prob) == base_pomdpstate.EventOutcome.BLOCK
         assert cost == pytest.approx(10e5, abs=0.1)
      if prob ==1.0:
         assert state.history.get_action_outcome(action, start, 1.0-prob) == base_pomdpstate.EventOutcome.TRAV
         assert cost == pytest.approx(4.0, abs=0.1)   

   best_action, cost = core.po_mcts(initial_state, n_iterations=2000)
   assert best_action == 4
   assert cost == pytest.approx(7.4, abs=0.1)

def test_sctpbase_state_disjointgraph_probs():

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
   best_action, cost = core.po_mcts(initial_state, n_iterations=1000)
   assert best_action == 2
   # assert cost == pytest.approx(expected_path123, abs=100.1)


def test_sctpbase_state_sgraph():
   start, goal, nodes, edges, robots = graphs.s_graph_unc()

   for edge in edges:
      edge.block_status = 0
      edge.block_prob = 0.1
   
   # for edge in edges:
   #    print(f"edge {edge.id} block status {edge.block_status} block prob {edge.block_prob} and cost {edge.cost}")

   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs,
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 2

   action = all_actions[0]
   # print(f"action {action}")
   assert action == 2 or action == 3
   best_action, exp_cost = core.po_mcts(initial_state, n_iterations=1000)

   assert best_action == 3
   # assert exp_cost == pytest.approx(8.0, abs=0.2)
   assert exp_cost == pytest.approx(0.9*8.0+0.2*1e6, abs=10.0)
