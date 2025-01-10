import pytest
from pouct_planner import core
from sctp import graphs, pomdp_state
from sctp import planning_loop as pl

def test_sctpbase_simplegraph():
   start = 1
   goal = 7
   nodes, edges = graphs.simple_graph()
   robots = graphs.RobotData(robot_id = 1, position=(1.0, 1.0), cur_vertex=start)
   
   # find_path, exe_path, cost = pl.sctpbase_pomcp_navigating(nodes, edges, robots, start, goal)
   # if find_path:
   #     print(f"find path: {exe_path} with cost: {cost}")

   name = "Simple Graph"
   graphs.plot_street_graph(nodes, edges, name)

   assert 3 == 3


def test_sctpbase_lineargraph():
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
   initial_state = pomdp_state.SCTPBaseState(edge_probs=edge_probs, 
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   all_actions = initial_state.get_actions()
   assert len(all_actions) == 1
   action = all_actions[0]
   assert action == 2

   outcome_states = initial_state.transition(action)
   assert len(outcome_states) == 2
   for state, (prob, cost) in outcome_states.items():
      print(f'prob: {prob}, cost: {cost}')
   


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

   best_action, cost = core.po_mcts(initial_state, n_iterations=2000)
   assert best_action == 2
   print(f"The cost is {cost}")
   assert cost == pytest.approx(15.0, abs=0.1)


def test_sctpbase_disjointgraph_simple():
   start = 1
   goal = 7
   nodes, edges = graphs.simple_disjoint_graph()
   robots = graphs.RobotData(robot_id = 1, position=(1.0, 1.0), cur_vertex=start)
   
   # find_path, exe_path, cost = pl.sctpbase_pomcp_navigating(nodes, edges, robots, start, goal)
   # if find_path:
   #     print(f"find path: {exe_path} with cost: {cost}")

   # name = "Simple Disjoint Graph"
   # graphs.plot_street_graph(nodes, edges, name)

if __name__ == "__main__":
    # test_sctpbase_simplegraph()
    test_sctpbase_lineargraph()