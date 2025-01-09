import pytest
from pouct_planner import core
from pouct_planner import graphs
from pouct_planner import planning_loop as pl

def test_sctpbase_simplegraph():
    nodes = None 
    edges = None
    start = 1
    goal = 1
    # nodes, edges = disjoint_graph()
    # name = "Disjoint Graph"
   #  print_graph(nodes, edges)
    # nodes, edges = disjoint_graph()
    # name = "Disjoint Graph"
   #  print_graph(nodes, edges)
    # nodes, edges = disjoint_graph()
    # name = "Disjoint Graph"
   #  print_graph(nodes, edges)
   # if graph_type == "simple_graph":
    start = 1
    goal = 7
    nodes, edges = graphs.simple_graph()
    robots = graphs.RobotData(robot_id = 1, coord_x = 1.0, coord_y = 1.0, cur_vertex=start)
    

    # find_path, exe_path, cost = pl.sctpbase_pomcp_navigating(nodes, edges, robots, start, goal)
    # if find_path:
    #     print(f"find path: {exe_path} with cost: {cost}")

    name = "Simple Graph"
    graphs.plot_street_graph(nodes, edges, name)

    assert 3 == 3

if __name__ == "__main__":
    test_sctpbase_simplegraph()