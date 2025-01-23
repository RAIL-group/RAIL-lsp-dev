import pytest
from sctp import base_pomdpstate, base_navigation
from pouct_planner import core
from sctp import graphs


def test_sctpbase_planner_lineargraph():
    start_node, goal_node, graph, robot = graphs.linear_graph_unc()
    
    graph.edges[0].block_prob = 0.0
    graph.edges[1].block_prob = 0.0
    initial_state = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal_node.id, robots=robot)
    all_actions = initial_state.get_actions()
    assert len(all_actions) == 1
    action = all_actions[0]
    assert action.end == 2
        
    best_action, cost, _  = core.po_mcts(initial_state, n_iterations=1000)
    desired_action = base_pomdpstate.Action(start_node=start_node.id, target_node=graph.vertices[1].id)
    assert best_action == desired_action
    assert cost == pytest.approx(15.0, abs=0.2)

    graph.edges[0].block_prob = 0.4
    graph.edges[1].block_prob = 0.9
    initial_state2 = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal_node.id, robots=robot)        
    best_action, cost, _  = core.po_mcts(initial_state2, n_iterations=1000)
    desired_action = base_pomdpstate.Action(start_node=start_node.id, target_node=graph.vertices[1].id)
    assert best_action == desired_action
    assert cost == pytest.approx(5.0*0.6 + (0.4+0.9)*base_pomdpstate.BLOCK_COST + 0.1*10 , abs=2.5)



def test_sctpbase_planner_disjoint():
    start, goal, graph, robots = graphs.disjoint_unc()
    for edge in graph.edges:
        edge.block_prob = 0.0
    init_state1 = base_pomdpstate.SCTPBaseState(graph=graph,goal=goal.id, robots=robots)
    all_actions = init_state1.get_actions()
    assert len(all_actions) == 2
    
    action2 = base_pomdpstate.Action(start_node=start.id, target_node=graph.vertices[1].id)
    action4 = base_pomdpstate.Action(start_node=start.id, target_node=graph.vertices[3].id)
    for action in all_actions:
        assert action in [action2, action4]
    action = all_actions[0] # action 2
    outcome_states = init_state1.transition(action)
    assert len(outcome_states) == 1

    for state, (prob, cost) in outcome_states.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV
        assert prob == 1.0
        assert cost == pytest.approx(4.0, abs=0.1)
    
    assert init_state1.history.get_data_length()==0
    best_action, cost, path  = core.po_mcts(init_state1, n_iterations=2000)
    print(best_action)
    assert best_action == action2
    assert cost == pytest.approx(8.0, abs=0.1)
    for p in path:
        print(p)
    # assert path[1] == 2 and path[2] == 3

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
   best_action, cost, path  = core.po_mcts(initial_state, n_iterations=2000, C=40.0)
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
   best_action, exp_cost, path  = core.po_mcts(initial_state, n_iterations=1000)

   assert best_action == 3
   print(f"The expected cost is {exp_cost}")
   assert path[1] == 3
   assert path[2] == 4
   # assert exp_cost == pytest.approx(8.0, abs=0.1)

def test_sctpbase_state_sgraph_prob1():
   start, goal, nodes, edges, robots = graphs.s_graph_unc() # edge 3-4 is blocked
   C = 55.0
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs,
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 2

   action = all_actions[0]
   assert action == 2 or action == 3
   best_action, exp_cost, path  = core.po_mcts(initial_state, C=C, n_iterations=10000)
   print(f"The expected cost is {exp_cost}")
   assert best_action == 2
   assert len(path) == 3
   assert path[1] == 2
   assert path[2] == 4

   # assert exp_cost == pytest.approx(0.9*8.0+0.2*1e6, abs=10.0)

def test_sctpbase_state_sgraph_prob2():
   C = 55.0
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
   best_action, exp_cost, path  = core.po_mcts(initial_state, C=C, n_iterations=10000)

   assert best_action == 3
   print(f"The expected cost is {exp_cost}")
   assert path[1] == 3
   assert path[2] == 2
   assert path[3] == 4

def test_sctpbase_state_mgraph_noblock():
   start, goal, nodes, edges, robots = graphs.m_graph_unc()
   C=5.5
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
   best_action, exp_cost, path  = core.po_mcts(initial_state, C=C, n_iterations=100000)

   assert best_action == 3
   assert path[1] == 3
   assert path[2] == 5
   assert path[3] == 7
   # assert exp_cost == pytest.approx(12.07, abs=0.1)

def test_sctpbase_state_mgraph_prob():
   start, goal, nodes, edges, robots = graphs.m_graph_unc()
   C=9.0

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
   best_action, exp_cost, path = core.po_mcts(initial_state, C=C, n_iterations=200000)
   print(f"The expected cost is {exp_cost}")
   assert best_action == 3
   assert path[1] == 3
   assert path[2] == 5
   assert path[3] == 6
   assert path[4] == 7
