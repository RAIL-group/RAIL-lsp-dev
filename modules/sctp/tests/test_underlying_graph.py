# test_explore_graph_planning.py
import matplotlib
# matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
import numpy as np
import sctp.sctp_graphs as graphs
import sctp.utils.underlying_graph as ug
from sctp.utils.underlying_graph import (
    ProbabilisticGraph,
    get_vertex_positions,
    create_adj_prob_matrices,
    set_edge_probabilities,
    get_initial_edges,
    sample_graph,
    compute_shortest_path_length as compute_shortest_path,
    plot_probabilistic_graph_with_samples,
)


def test_probabilistic_graph_dataclass():
    """Test that ProbabilisticGraph dataclass exists with correct fields."""
    starts, goals, graph = graphs.disjoint_unc()
    graph.print_graph_config()
    edges = graphs.get_initial_edges(graph)
    positions = get_vertex_positions(graph.vertices)
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
    positions = get_vertex_positions(graph.vertices)
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
    positions = get_vertex_positions(graph.vertices)
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
    positions = get_vertex_positions(graph.vertices)
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
    positions = get_vertex_positions(graph.vertices)
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


def make_inputs(prob_list, adj_list, n):
    """Helper to build square matrices from flat lists."""
    return (np.array(prob_list, dtype=float).reshape(n, n),
            np.array(adj_list, dtype=float).reshape(n, n))


# class TestGetOptimisticAdjMatrix:

def test_no_blocked_edges():
    """All probs < 1.0 → full graph preserved."""
    prob, adj = make_inputs(
        [0,   0.2, 0.5,
            0.2, 0,   0.3,
            0.5, 0.3, 0  ],
        [0, 1, 2,
            1, 0, 3,
            2, 3, 0], n=3
    )
    result = ug.get_optimistic_adj_matrix(prob, adj)
    np.testing.assert_array_equal(result, adj)

def test_all_blocked_edges():
    """All probs == 1.0 → empty graph (all zeros)."""
    prob, adj = make_inputs(
        [1, 1, 1,
            1, 1, 1,
            1, 1, 1],
        [0, 5, 9,
            5, 0, 3,
            9, 3, 0], n=3
    )
    result = ug.get_optimistic_adj_matrix(prob, adj)
    np.testing.assert_array_equal(result, np.zeros((3, 3)))

def test_some_blocked_edges():
    """Only edges with prob == 1.0 are removed."""
    prob, adj = make_inputs(
        [0,   1.0, 0.5,
            1.0, 0,   0.0,
            0.5, 0.0, 0  ],
        [0, 10, 20,
            10, 0,  30,
            20, 30,  0], n=3
    )
    result = ug.get_optimistic_adj_matrix(prob, adj)
    expected = np.array([
        [0,  0, 20],
        [0,  0, 30],
        [20, 30,  0]
    ], dtype=float)
    np.testing.assert_array_equal(result, expected)

def test_zero_prob_edge_kept():
    """prob == 0.0 means definitely passable → edge must be kept."""
    prob, adj = make_inputs(
        [0,   0.0,
            0.0, 0  ],
        [0, 7,
            7, 0], n=2
    )
    result = ug.get_optimistic_adj_matrix(prob, adj)
    np.testing.assert_array_equal(result, adj)

def test_symmetry():
    """Output matrix must always be symmetric."""
    rng = np.random.default_rng(42)
    n = 5
    prob = np.triu(rng.uniform(0, 1, (n, n)), k=1)
    prob = prob + prob.T                       # symmetric prob matrix
    adj  = np.triu(rng.integers(1, 20, (n, n)).astype(float), k=1)
    adj  = adj + adj.T

    result = ug.get_optimistic_adj_matrix(prob, adj)
    np.testing.assert_array_equal(result, result.T)

def test_diagonal_untouched():
    """Diagonal should remain 0 (self-loops ignored by triu k=1)."""
    prob, adj = make_inputs(
        [0.5, 0.5,
            0.5, 0.5],
        [99, 1,
            1, 99], n=2
    )
    result = ug.get_optimistic_adj_matrix(prob, adj)
    assert result[0, 0] == 0
    assert result[1, 1] == 0

def test_single_vertex():
    """1×1 graph edge case should return a 1×1 zero matrix."""
    prob = np.array([[0.0]])
    adj  = np.array([[0.0]])
    result = ug.get_optimistic_adj_matrix(prob, adj)
    np.testing.assert_array_equal(result, np.zeros((1, 1)))
