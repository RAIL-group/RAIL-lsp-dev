import copy
import pytest
from sctp import graphs


def test_graph_vertex_check():
    start = graphs.Vertex(coords=(0.0, 0.0))
    goal = graphs.Vertex(coords=(15.0, 0.0))
    node1 = graphs.Vertex(coords=(5.0, 0.0))

    graph = graphs.Graph(vertices=[start, goal, node1])
    graph.add_edge(start, node1, 0.0)
    graph.add_edge(node1, goal, 0.0)

    assert start in graph.vertices
    assert goal in graph.vertices

    assert start != goal  # Start should not be equal to goal
    assert start == start  # Start should be equal to itself

    start_copy = copy.copy(start)
    assert start_copy == start  # Still the same because id and coords are the same

    new_start = graphs.Vertex(coords=(0.0, 0.0))
    assert new_start != start  # Although coords are same, id is different
