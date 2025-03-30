import copy, random
import pytest
import numpy as np
from sctp import sctp_graphs as graphs
from sctp.utils import plotting


def test_sctp_graph_vertex_edge_check():
    start_node = graphs.Vertex(coord=(0.0, 0.0))
    node1 = graphs.Vertex(coord=(5.0, 0.0))
    goal_node = graphs.Vertex(coord=(15.0, 0.0))
    
    graph = graphs.Graph(vertices=[start_node, goal_node, node1])
    graph.add_edge(start_node, node1, 0.0)
    graph.add_edge(node1, goal_node, 0.0)
    assert start_node in graph.vertices
    assert goal_node in graph.vertices

    assert start_node != goal_node  # Start should not be equal to goal
    assert start_node == start_node  # Start should be equal to itself
    assert len(graph.vertices) == 3
    assert len(graph.edges) == 4
    assert len(graph.pois) == 2
    for vertex in graph.vertices:
        assert vertex.block_prob == 0.0

    for poi in graph.pois:
        assert poi.block_prob == 0.0

    start_copy = copy.copy(start_node)
    assert start_copy == start_node  # Still the same because id and coords are the same

    new_start = graphs.Vertex(coord=(0.0, 0.0))
    assert new_start != start_node  # Although coords are same, id is different

    # edge test
    edge = graph.edges[0]
    new_edge = copy.copy(edge)
    assert edge == new_edge
    # edge with opposite direction
    opp_edge = graphs.Edge(edge.v2, edge.v1)
    assert opp_edge == edge
    vertices = graph.vertices + graph.pois
    graphs.dijkstra(vertices, graph.edges, goal=goal_node)
    for vertex in graph.vertices:
        if vertex.id == 1:
            assert vertex.heur2goal == 15.0
        elif vertex.id == 2:
            assert vertex.heur2goal == 10.0
        elif vertex.id == 3:
            assert vertex.heur2goal == 0.0
    for poi in graph.pois:
        if poi.id == 4:
            assert poi.heur2goal == 12.5
        elif poi.id == 5:
            assert poi.heur2goal == 5.0
    # graphs.plot_sctpgraph(vertices, graph.edges, startID=start_node.id, goalID=goal_node.id)

def test_sctp_graph_linear_check():
    start, goal, ln_graph = graphs.linear_graph_unc()
    
    assert start in ln_graph.vertices
    assert goal in ln_graph.vertices
    assert len(ln_graph.vertices) == 3
    assert len(ln_graph.edges) == 4

    for vertex in ln_graph.vertices:
        if vertex.id == 1:
            assert vertex.heur2goal == 15.0
        elif vertex.id == 2:
            assert vertex.heur2goal == 10.0
        elif vertex.id == 3:
            assert vertex.heur2goal == 0.0

    for poi in ln_graph.pois:
        assert len(poi.neighbors) == 2
        if poi.id == 4:
            assert poi.heur2goal == 12.5
        elif poi.id == 5:
            assert poi.heur2goal == 5.0
    for edge in ln_graph.edges:
        assert edge.v1.block_prob == 0.0 or edge.v2.block_prob == 0.0

def test_sctp_graph_disjoint_check():
    start, goal, dj_graph = graphs.disjoint_unc()
    
    assert start in dj_graph.vertices
    assert goal in dj_graph.vertices
    assert len(dj_graph.vertices) == 4
    assert len(dj_graph.edges) == 8
    for vertex in dj_graph.vertices:
        assert vertex.block_prob == 0.0
        if vertex.id == 1:
            assert vertex.heur2goal == 8.0
        elif vertex.id == 2:
            assert vertex.heur2goal == 4.0
        elif vertex.id == 3:
            assert vertex.heur2goal == 0.0
        elif vertex.id == 4:
            assert vertex.heur2goal == pytest.approx(5.56, 0.1)
    for poi in dj_graph.pois:
        assert len(poi.neighbors) == 2
        if poi.id == 5:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(6.0, 0.1)
        elif poi.id == 6:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(2.78, 0.1)
        elif poi.id == 7:
            assert poi.block_prob == 0.9
            assert poi.heur2goal == 2.0
        elif poi.id == 8:
            assert poi.block_prob == 0.2
            assert poi.heur2goal == pytest.approx(2.78+5.56, 0.1)
    for edge in dj_graph.edges:
        assert edge.v1.block_prob == 0.0 or edge.v2.block_prob == 0.0

def test_sctp_graph_simple_check():
    start, goal, s_graph = graphs.s_graph_unc()
    
    assert start in s_graph.vertices
    assert goal in s_graph.vertices

    assert len(s_graph.vertices) == 4
    assert len(s_graph.edges) == 10
    for vertex in s_graph.vertices:
        assert vertex.block_prob == 0.0
        if vertex.id == 1:
            assert vertex.heur2goal == 8.0
        elif vertex.id == 2:
            assert vertex.heur2goal == pytest.approx(5.56, 0.1)
        elif vertex.id == 3:
            assert vertex.heur2goal == 4.0
        elif vertex.id == 4:
            assert vertex.heur2goal == 0.0
    for poi in s_graph.pois:
        assert len(poi.neighbors) == 2
        if poi.id == 5:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(5.56+2.78, 0.1)
        elif poi.id == 6:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(6.0, 0.1)
        elif poi.id == 7:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == 6.0
        elif poi.id == 8:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(2.78, 0.1)
        elif poi.id == 9:
            assert poi.block_prob == 0.9
            assert poi.heur2goal == pytest.approx(2.0, 0.1)
    for edge in s_graph.edges:
        assert edge.v1.block_prob == 0.0 or edge.v2.block_prob == 0.0
    

def test_sctp_graph_medium_check():
    start, goal, m_graph = graphs.m_graph_unc()
    
    assert start in m_graph.vertices
    assert goal in m_graph.vertices

    assert len(m_graph.vertices) == 7
    assert len(m_graph.edges) == 24
    assert len(m_graph.pois) == 12

    # assert robots[0].cur_node == start.id
    # assert robots[0].cur_pose[0] == start.coord[0]
    # assert robots[0].cur_pose[1] == start.coord[1]
    for vertex in m_graph.vertices:
        if vertex.id == 1:
            assert vertex.heur2goal == pytest.approx(4.0+4.5+3.6, 0.2)
        elif vertex.id == 2:
            assert vertex.heur2goal == pytest.approx(23.3, 0.2)
        elif vertex.id == 3:
            assert vertex.heur2goal == pytest.approx(8.5, 0.1)
        elif vertex.id == 4:
            assert vertex.heur2goal == pytest.approx(5.7, 0.05)
        elif vertex.id == 5:
            assert vertex.heur2goal == pytest.approx(4.0, 0.05)
        elif vertex.id == 6:
            assert vertex.heur2goal == pytest.approx(5.7, 0.05)
        elif vertex.id == 7:
            assert vertex.heur2goal == 0.0
        
    for poi in m_graph.pois:
        assert len(poi.neighbors) == 2
        if poi.id == 8:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(0.5*12.5+3.6+4.5+4.0, 0.1)
        elif poi.id == 9:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(0.5*3.6+4.5+4.0, 0.1)
        elif poi.id == 10:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(0.5*16.0+4.5+4.0, 0.1)
        elif poi.id == 11:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(0.5*19.3+4.0, 0.1)
        elif poi.id == 12:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(0.5*19.0+5.7, 0.1)
        elif poi.id == 13:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(0.5*4.5+5.7, 0.1)
        elif poi.id == 14:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(0.5*4.5+4.0, 0.1)
        elif poi.id == 15:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(0.5*4.0+4.0, 0.1)
        elif poi.id == 16:
            assert poi.block_prob == 0.9
            assert poi.heur2goal == pytest.approx(0.5*5.7, 0.1)
        elif poi.id == 17:
            assert poi.block_prob == 0.9
            assert poi.heur2goal == pytest.approx(0.5*4.0, 0.1)
        elif poi.id == 18:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(0.5*5.7, 0.1)
        elif poi.id == 19:
            assert poi.block_prob == 0.1
            assert poi.heur2goal == pytest.approx(0.5*4.0+4.0, 0.1)

    for edge in m_graph.edges:
        assert edge.v1.block_prob == 0.0 or edge.v2.block_prob == 0.0


def test_sctp_rangraph_unblockPath():
    seed = np.random.randint(1000,2000)
    np.random.seed(seed)
    random.seed(seed)
    start, goal, ran_graph = graphs.random_graph(n_vertex=8)
    plotting.plot_sctpgraph(graph=ran_graph, startID=start.id, goalID=goal.id, seed=seed)

   
