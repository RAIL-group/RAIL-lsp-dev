# test_explore_graph_planning.py
import matplotlib
# matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
import numpy as np
import sctp.sctp_graphs as graphs
from sctp.utils.underlying_graph import (
    ProbabilisticGraph,
    generate_vertices,
    create_adj_prob_matrices,
    set_edge_probabilities,
    get_initial_edges,
    sample_graph,
    compute_shortest_path,
    plot_probabilistic_graph_with_samples,
)


def test_probabilistic_graph_dataclass():
    """Test that ProbabilisticGraph dataclass exists with correct fields."""
    starts, goals, graph = graphs.disjoint_unc()
    graph.print_graph_config()
    edges = graphs.get_initial_edges(graph)
    positions = generate_vertices(graph.vertices)
    adjacency, probabilities = create_adj_prob_matrices(edges, positions)

    pg = ProbabilisticGraph(
        positions=positions,
        adjacency=adjacency,
        probabilities=probabilities,
    )
    print(pg.positions)
    print(pg.adjacency)
    print(pg.probabilities)
    assert pg.positions.shape == (4, 2)
    assert pg.adjacency.shape == (4, 4)
    assert pg.probabilities.shape == (4, 4)
    neighbor_vertex1 = probabilities[0][probabilities[0] > 0]
    assert len(neighbor_vertex1) == len(graph.vertices[0].neighbors)    
    np.testing.assert_array_equal(pg.adjacency, pg.adjacency.T)
    np.testing.assert_array_equal(pg.probabilities, pg.probabilities.T)
    np.testing.assert_array_equal(np.diag(pg.adjacency), 0)



def test_set_edges():
    """Test that the function set edges """
    starts, goals, graph = graphs.disjoint_unc()
    graph.print_graph_config()
    edges = get_initial_edges(graph)
    positions = generate_vertices(graph.vertices)
    adjacency, probabilities = create_adj_prob_matrices(edges, positions)

    pg = ProbabilisticGraph(
        positions=positions,
        adjacency=adjacency,
        probabilities=probabilities,
    )
    edges = [(0, 1), (2, 3)]
    probs = np.array([0.0, 1.0])
    probabilities_set = set_edge_probabilities(probs=probs, edges=edges, probabilities=pg.probabilities)
    assert probabilities_set[0,1] == 0.0
    assert probabilities_set[1,0] == 0.0
    assert probabilities_set[2,3] == 1.0
    assert probabilities_set[3,2] == 1.0

def test_sample_graph():
    """Test that the function set edges """
    starts, goals, graph = graphs.disjoint_unc()
    graph.print_graph_config()
    edges = get_initial_edges(graph)
    positions = generate_vertices(graph.vertices)
    adjacency, probabilities = create_adj_prob_matrices(edges, positions)

    pg = ProbabilisticGraph(
        positions=positions,
        adjacency=adjacency,
        probabilities=probabilities,
    )
    edges = [(0, 1), (2, 3)]
    probs = np.array([0.0, 1.0])
    sample = sample_graph(prob_graph=pg, probs=probs, edges=edges)
    assert sample[2,3] == 0.0
    print(sample)

def test_shortest_path_sgraph():
    """Test that the function set edges """
    print("")
    starts, goals, graph = graphs.s_graph_unc()
    graph.print_graph_config()
    edges = get_initial_edges(graph)
    positions = generate_vertices(graph.vertices)
    adjacency, probabilities = create_adj_prob_matrices(edges, positions)

    pg = ProbabilisticGraph(
        positions=positions,
        adjacency=adjacency,
        probabilities=probabilities,
    )
    edges = [(0, 1), (2, 3)]
    probs = np.array([0.0, 1.0])
    sample = sample_graph(prob_graph=pg, probs=probs, edges=edges)
    assert sample[2,3] == 0.0
    cost = compute_shortest_path(sample, start=0, end=3)
    print(sample)
    print(cost)

def test_shortest_path_island_bridges_graph():
    """Test that the function set edges """
    print("")
    starts, goals, graph = graphs.get_insland_bridges_graph()
    graph.print_graph_config()
    edges = get_initial_edges(graph)
    positions = generate_vertices(graph.vertices)
    adjacency, probabilities = create_adj_prob_matrices(edges, positions)

    pg = ProbabilisticGraph(
        positions=positions,
        adjacency=adjacency,
        probabilities=probabilities,
    )
    edges = [(0, 1), (2, 3)]
    probs = np.array([0.0, 1.0])
    new_probabilities = set_edge_probabilities(probs=probs, edges=edges, probabilities=pg.probabilities)
    sample = sample_graph(prob_graph=pg, probs=new_probabilities)
    assert sample[2,3] == 0.0
    cost = compute_shortest_path(sample, start=0, end=3)
    print(sample)
    print(cost)

# def test_create_probabilistic_graph():
#     """Test the convenience function creates a valid ProbabilisticGraph."""
#     pg = create_probabilistic_graph(num_vertices=50, threshold=30.0)

#     assert isinstance(pg, ProbabilisticGraph)
#     assert pg.positions.shape == (50, 2)
#     assert pg.adjacency.shape == (50, 50)
#     assert pg.probabilities.shape == (50, 50)

#     # Verify symmetry
#     np.testing.assert_array_equal(pg.adjacency, pg.adjacency.T)
#     np.testing.assert_array_equal(pg.probabilities, pg.probabilities.T)


# def test_sample_graph_symmetry():
#     """Test that sampled graph is symmetric."""
#     pg = create_probabilistic_graph(num_vertices=50, threshold=30.0)
#     sampled = sample_graph(pg)
#     np.testing.assert_array_equal(sampled, sampled.T)


# def test_sample_graph_subset():
#     """Test that sampled edges are a subset of original adjacency."""
#     pg = create_probabilistic_graph(num_vertices=50, threshold=30.0)
#     sampled = sample_graph(pg)

#     # Sampled should only have edges where adjacency has edges
#     assert np.all((sampled > 0) <= (pg.adjacency > 0))

#     # Where sampled is non-zero, it should equal adjacency
#     nonzero_mask = sampled > 0
#     np.testing.assert_array_equal(sampled[nonzero_mask], pg.adjacency[nonzero_mask])


# def test_sample_graph_respects_probability_one():
#     """Test that edges with probability 1.0 always exist."""
#     positions = np.array([[0, 0], [3, 4]])
#     adjacency = np.array([[0, 5.0], [5.0, 0]])
#     probabilities = np.array([[0, 1.0], [1.0, 0]])  # Probability 1.0
#     pg = ProbabilisticGraph(positions=positions, adjacency=adjacency, probabilities=probabilities)

#     # Sample many times - edge should always exist
#     for _ in range(10):
#         sampled = sample_graph(pg)
#         assert sampled[0, 1] == 5.0
#         assert sampled[1, 0] == 5.0


# def test_sample_graph_respects_probability_zero():
#     """Test that edges with probability 0.0 never exist."""
#     positions = np.array([[0, 0], [3, 4]])
#     adjacency = np.array([[0, 5.0], [5.0, 0]])
#     probabilities = np.array([[0, 0.0], [0.0, 0]])  # Probability 0.0
#     pg = ProbabilisticGraph(positions=positions, adjacency=adjacency, probabilities=probabilities)

#     sampled = sample_graph(pg)
#     assert sampled[0, 1] == 0.0
#     assert sampled[1, 0] == 0.0


# def test_shortest_path_known_graph():
#     """Test shortest path on a simple known graph."""
#     # Triangle: 0 -- 1 -- 2, with direct edge 0 -- 2
#     # Edge weights: 0-1: 1.0, 1-2: 1.0, 0-2: 3.0
#     # Shortest 0->2 should be 2.0 (via 1), not 3.0 (direct)
#     adjacency = np.array([
#         [0, 1.0, 3.0],
#         [1.0, 0, 1.0],
#         [3.0, 1.0, 0],
#     ])

#     path_cost = compute_shortest_path(adjacency, start=0, end=2)
#     assert path_cost == 2.0


# def test_shortest_path_no_path():
#     """Test that None is returned when no path exists."""
#     # Two disconnected vertices
#     adjacency = np.array([
#         [0, 0],
#         [0, 0],
#     ])

#     path_cost = compute_shortest_path(adjacency, start=0, end=1)
#     assert path_cost is None


# def test_shortest_path_negative_index():
#     """Test that negative index works for end vertex."""
#     adjacency = np.array([
#         [0, 1.0, 0],
#         [1.0, 0, 2.0],
#         [0, 2.0, 0],
#     ])

#     # end=-1 should be vertex 2
#     path_cost = compute_shortest_path(adjacency, start=0, end=-1)
#     assert path_cost == 3.0  # 0 -> 1 -> 2


# def test_plot_returns_figure():
#     """Test that plotting function returns a matplotlib Figure."""
#     pg = create_probabilistic_graph(num_vertices=20, threshold=40.0)
#     fig = plot_probabilistic_graph_with_samples(pg, num_samples=3, seed=42)

#     assert isinstance(fig, plt.Figure)
#     assert len(fig.axes) == 5  # 2x2 grid + colorbar
#     plt.close(fig)
