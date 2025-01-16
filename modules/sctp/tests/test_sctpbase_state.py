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

   best_action, cost = core.po_mcts(initial_state, n_iterations=10000)
   assert best_action == 2
   assert cost == pytest.approx(15.0, abs=0.1)


def test_sctpbase_state_disjoint_noblock():
   start, goal, nodes, edges, robots = graphs.disjoint_unc()
   for edge in edges:
      edge.block_status = 0
      edge.block_prob = 0.0
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs,
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 2
   action = all_actions[0]
   assert action == 2 or action == 4

   outcome_states = initial_state.transition(action)
   assert len(outcome_states) == 1
   for state, (prob, cost) in outcome_states.items():
      if prob ==1.0 and state.history.get_action_outcome(action, start, 1.0-prob) == base_pomdpstate.EventOutcome.TRAV:
         assert cost == pytest.approx(4.0, abs=0.1)   

   best_action, cost = core.po_mcts(initial_state, n_iterations=2000)
   assert best_action == 2
   assert cost == pytest.approx(8.0, abs=0.1)

def test_sctpbase_state_disjoint_prob():
   start, goal, nodes, edges, robots = graphs.disjoint_unc()

   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs,
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 2
   action = all_actions[0]
   assert action == 2 or action == 4
   best_action, cost = core.po_mcts(initial_state, n_iterations=2000, C=40.0)
   assert best_action == 4
   # assert cost == pytest.approx(0.2*1e6+0.8*5.66 + 0.15*1e6+0.85*5.66, abs=1000.1)


def test_sctpbase_state_sgraph_noblock():
   start, goal, nodes, edges, robots = graphs.s_graph_unc()
   for edge in edges:
      edge.block_status = 0
      edge.block_prob = 0.0
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs,
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 2

   action = all_actions[0]
   assert action == 2 or action == 3
   best_action, exp_cost = core.po_mcts(initial_state, n_iterations=1000)

   assert best_action == 3
   assert exp_cost == pytest.approx(8.0, abs=0.1)

def test_sctpbase_state_sgraph_prob1():
   start, goal, nodes, edges, robots = graphs.s_graph_unc()
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs,
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 2

   action = all_actions[0]
   assert action == 2 or action == 3
   best_action, exp_cost = core.po_mcts(initial_state, n_iterations=1000)

   assert best_action == 2
   # assert exp_cost == pytest.approx(8.0, abs=0.2)
   # assert exp_cost == pytest.approx(0.9*8.0+0.2*1e6, abs=10.0)

def test_sctpbase_state_sgraph_prob2():
   C = 80.0
   start, goal, nodes, edges, robots = graphs.s_graph_unc()
   for edge in edges: # edge 1-2 and 3-4 are blocked
      if edge.id == (1,2):
         # print(f"edge {edge.id} is blocked")
         edge.block_status = 1
         edge.block_prob = 0.8
   
   # for edge in edges:
   #    print(f"edge {edge.id} with block prob {edge.block_prob}, block status {edge.block_status} and cost {edge.cost}")
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs,
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 2

   action = all_actions[0]
   assert action == 2 or action == 3
   best_action, exp_cost = core.po_mcts(initial_state, C=C, n_iterations=20000)

   assert best_action == 3

def test_sctpbase_state_mgraph_noblock():
   start, goal, nodes, edges, robots = graphs.m_graph_unc()
   C=30.0
   for edge in edges: # edge 1-2 and 3-4 are blocked
      edge.block_status = 0
      edge.block_prob = 0.0
   
   # for edge in edges:
   #    print(f"edge {edge.id} with block prob {edge.block_prob}, block status {edge.block_status} and cost {edge.cost}")
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs,
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 2

   action = all_actions[0]
   assert action == 2 or action == 3
   best_action, exp_cost = core.po_mcts(initial_state, C=C, n_iterations=10000)

   assert best_action == 3
   # assert exp_cost == pytest.approx(12.07, abs=0.1)

def test_sctpbase_state_mgraph_prob():
   start, goal, nodes, edges, robots = graphs.m_graph_unc()
   C=120.0
   # for edge in edges: # edge 1-2 and 3-4 are blocked
   #    edge.block_status = 0
   #    edge.block_prob = 0.0
   
   # for edge in edges:
   #    print(f"edge {edge.id} with block prob {edge.block_prob}, block status {edge.block_status} and cost {edge.cost}")
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs,
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 2

   action = all_actions[0]
   assert action == 2 or action == 3
   best_action, exp_cost = core.po_mcts(initial_state, C=C, n_iterations=80000)

   assert best_action == 3
