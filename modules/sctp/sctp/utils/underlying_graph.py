"""Probabilistic graph exploration with sampling and shortest path computation."""

from dataclasses import dataclass
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple
from sctp import param

@dataclass
class ProbabilisticGraph:
    """A graph with uncertain edge existence.

    Attributes:
        positions: (n, 2) array of vertex (x, y) coordinates
        adjacency: (n, n) symmetric matrix of edge costs (Euclidean distances)
        probabilities: (n, n) symmetric matrix of edge existence probabilities
    """
    positions: np.ndarray
    adjacency: np.ndarray
    probabilities: np.ndarray

def get_initial_edges(graph):
    # edges with 0-based indices for the underlying graph
    initial_edges = []
    for poi in graph.pois:
        initial_edges.append([poi.neighbors[0]-1, poi.neighbors[1]-1, poi.block_prob])
    return initial_edges

    
def set_edge_probabilities(probs: np.ndarray, edges: list, probabilities: np.ndarray):
    """Set edge probabilities for specified edges with given modified vertices' indices
    Args:
        prob: Probability to set for the edges.
        edges: List of (i, j) tuples specifying edges to update.
    """
    probabilities_copy = probabilities.copy()
    
    if len(edges) == 0:
        return probabilities_copy    
    edges_array = np.array(edges)
    i_indices = edges_array[:, 0]
    j_indices = edges_array[:, 1]
    probabilities_copy[i_indices, j_indices] = probs
    probabilities_copy[j_indices, i_indices] = probs    
    return probabilities_copy

def get_vertex_positions(vertices) -> np.ndarray:
    positions = np.array([[v.coord[0], v.coord[1]] for v in vertices])
    return positions


def create_adj_prob_matrices(edges, positions):
    """Create symmetric adjacency matrix with Euclidean distances with given modified vertices' indices
    Args:
        positions: (n, 2) array of vertex coordinates.
        threshold: Maximum distance for edge existence. Edges beyond this are 0.
    Returns:
        (n, n) symmetric matrix where entry (i,j) is the Euclidean distance
        between vertices i and j if <= threshold, else 0.
    """
    adjacency = np.zeros((positions.shape[0], positions.shape[0]), dtype=np.float32)
    probabilities = np.zeros((positions.shape[0], positions.shape[0]), dtype=np.float32)
    # Compute upper triangle only
    for edge in edges:
        dist = np.linalg.norm(np.array(positions[edge[0]]) - np.array(positions[edge[1]]))
        assert adjacency[edge[0], edge[1]] == 0.0
        adjacency[edge[0], edge[1]] = dist
        probabilities[edge[0], edge[1]] = edge[2]

    adjacency = adjacency + adjacency.T
    probabilities = probabilities + probabilities.T
    return adjacency, probabilities


def sample_graph(prob_graph: ProbabilisticGraph, probs: np.ndarray) -> np.ndarray:
    """Sample a concrete adjacency matrix from the probabilistic graph.
    Generates random values and thresholds against probabilities to determine
    which edges exist in this sample.
    Args:
        prob_graph: The probabilistic graph to sample from.

    Returns:
        (n, n) symmetric adjacency matrix with sampled edges.
    """
    n = prob_graph.adjacency.shape[0]

    # Generate random values for upper triangle
    random_vals = np.random.random((n, n))
    edge_exists = random_vals >= probs

    # Apply to adjacency, keep only upper triangle
    sampled = np.triu(prob_graph.adjacency * edge_exists, k=1)

    # Mirror to lower triangle
    return sampled + sampled.T

def get_certain_adj_matrix(prob_matrix: np.ndarray, adj_matrix: np.ndarray):
    n = prob_matrix.shape[0]

    # Generate random values for upper triangle
    random_vals = np.zeros((n, n))
    edge_exists = random_vals == prob_matrix

    # Apply to adjacency, keep only upper triangle
    sampled = np.triu(adj_matrix * edge_exists, k=1)
    return sampled + sampled.T
    


def compute_shortest_path_length(adjacency: np.ndarray, start: int=0, end: int=-1) -> float:
    """Compute shortest path cost between two vertices with given modified vertices' indices

    Args:
        adjacency: (n, n) adjacency matrix with edge weights.
        start: Starting vertex index.
        end: Ending vertex index (supports negative indexing).

    Returns:
        The shortest path cost, or None if no path exists.
    """
    n = adjacency.shape[0]

    # Handle negative indexing
    if end < 0:
        end = n + end

    # Create networkx graph from adjacency matrix
    G = nx.from_numpy_array(adjacency)

    try:
        return nx.shortest_path_length(G, source=start, target=end, weight="weight")
    except nx.NetworkXNoPath:
        return -1.0

def get_shortest_path(adjacency: np.ndarray, start: int = 0, end: int = -1) -> List[int]:
    """Compute shortest path between two vertices with given modified vertices' indices

    Args:
        adjacency: (n, n) adjacency matrix with edge weights.
        start: Starting vertex index.
        end: Ending vertex index (supports negative indexing).

    Returns:
        The shortest path or None if no path exists.
    """
    n = adjacency.shape[0]
    # Handle negative indexing
    if end < 0:
        end = n + end
    # Create networkx graph from adjacency matrix
    G = nx.from_numpy_array(adjacency)
    try:
        return nx.shortest_path(G, source=start, target=end)
    except nx.NetworkXNoPath:
        return []

def get_shortest_path_from_vertices(adj_matrix, start: int, targets: List[int]) -> List[int]:
    """Compute shortest path between two vertices with initial graph's vertices' indices

    Args:
        adjacency: (n, n) adjacency matrix with edge weights.
        start: Starting vertex index.
        end: Ending vertex index (supports negative indexing).

    Returns:
        The shortest path or None if no path exists with the graph's vertices' indices
    """
    return_cost = 0.0
    if start in targets:
        return [start], 0.0
    if len(targets) == 2:
        costs = []
        for target in targets:
            cost = compute_shortest_path_length(adj_matrix, start=start-1, end=target -1)
            costs.append(cost)
        if all (costs) < 0.0:
            return [], -1.0
        elif costs[0] < 0.0:
            path = get_shortest_path(adj_matrix, start=start-1, end=targets[1] -1)
            return_cost = costs[1]                    
        elif costs[1] < 0.0:
            path = get_shortest_path(adj_matrix, start=start-1, end=targets[0] -1)
            return_cost = costs[0]
        elif costs[0] <= costs[1]:
            path = get_shortest_path(adj_matrix, start=start-1, end=targets[0] -1)
            return_cost = costs[0]
        else:
            path = get_shortest_path(adj_matrix, start=start-1, end=targets[1] -1)
            return_cost = costs[1]
        return [p+1 for p in path], return_cost
    else: # reaching goal action
        assert len(targets) == 1
        cost = compute_shortest_path_length(adj_matrix, start=start-1, end=targets[0] -1)
        path = get_shortest_path(adj_matrix, start=start-1, end=targets[0] -1)
        return [p+1 for p in path] if len(path) > 0 else [], cost
    
def get_updated_prob_matrix(state):
    status_edges = []
    probs = []
    for act, outcome  in state.history.get_data().items():
        if act.target not in state.graph.poiIDs:
            continue
        neighbors = state.graph.get_poi(act.target).neighbors
        status_edges.append([neighbors[0]-1, neighbors[1]-1])
        if outcome == param.EventOutcome.BLOCK:
            probs.append(1.0)
        else:            
            probs.append(0.0)
    if len(status_edges) > 0:
        new_probabilities = set_edge_probabilities(probs=np.array(probs), edges=status_edges, \
                    probabilities=state.pg_probabilities) 
    else:
        new_probabilities = state.pg_probabilities.copy()
    return new_probabilities
    

def plot_probabilistic_graph_with_samples(
    prob_graph: ProbabilisticGraph,
    num_samples: int = 3,
    seed: int=0
) -> plt.Figure:
    """Create visualization of probabilistic graph and sampled instances.

    Args:
        prob_graph: The probabilistic graph to visualize.
        num_samples: Number of samples to show (default 3, displayed in remaining subplots).
        seed: Random seed for reproducible samples.

    Returns:
        A matplotlib Figure with 2x2 grid:
        - Top-left: Probabilistic graph (edge color = probability)
        - Other panels: Sampled graphs with shortest paths highlighted
    """
    if seed is not None:
        np.random.seed(seed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    positions = prob_graph.positions
    n = len(positions)

    # Color map for probabilities
    cmap = cm.viridis

    def draw_graph(ax, adjacency, title, show_probability=False, path_edges=None):
        """Helper to draw a graph on an axis."""
        ax.set_title(title)
        ax.set_aspect("equal")

        # Draw edges
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] > 0:
                    x_coords = [positions[i, 0], positions[j, 0]]
                    y_coords = [positions[i, 1], positions[j, 1]]

                    if show_probability:
                        prob = prob_graph.probabilities[i, j]
                        color = cmap(prob)
                        ax.plot(x_coords, y_coords, color=color, linewidth=1, alpha=0.7)
                    elif path_edges and (i, j) in path_edges:
                        ax.plot(x_coords, y_coords, color="red", linewidth=2.5, zorder=2)
                    else:
                        ax.plot(x_coords, y_coords, color="gray", linewidth=0.5, alpha=0.5)

        # Draw vertices
        ax.scatter(positions[:, 0], positions[:, 1], c="blue", s=20, zorder=3)

        # Highlight start and end
        ax.scatter([positions[0, 0]], [positions[0, 1]], c="green", s=100, marker="^", zorder=4, label="Start")
        ax.scatter([positions[-1, 0]], [positions[-1, 1]], c="red", s=100, marker="s", zorder=4, label="End")
        ax.legend(loc="upper right")

    # Plot probabilistic graph
    draw_graph(axes[0], prob_graph.adjacency, "Probabilistic Graph (color=probability)", show_probability=True)

    # Add colorbar for probabilistic graph
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.7, vmax=1.0))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[0], label="Edge Probability")

    # Plot sampled graphs
    for idx in range(num_samples):
        ax = axes[idx + 1]
        sampled = sample_graph(prob_graph)
        path_cost = compute_shortest_path_length(sampled, start=0, end=-1)

        # Find path edges for highlighting
        path_edges = set()
        if path_cost is not None:
            # Reconstruct path using networkx
            G = nx.from_numpy_array(sampled)
            try:
                path = nx.shortest_path(G, source=0, target=n - 1, weight="weight")
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    path_edges.add((min(a, b), max(a, b)))
            except nx.NetworkXNoPath:
                pass

        title = f"Sample {idx + 1}: "
        title += f"Path Cost = {path_cost:.2f}" if path_cost else "No Path"
        draw_graph(ax, sampled, title, path_edges=path_edges)

    plt.tight_layout()
    return fig
