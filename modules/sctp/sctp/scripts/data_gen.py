import os
# import lsp
# import torch
import random, argparse
import numpy as np
# import pickle
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import random
import sctp.sctp_graphs as graphs
import sctp.utils.underlying_graph as ug
import sctp.action_estimation as ae
# from torch_geometric.data import Data
import learning


@dataclass
class GraphData:
    """Data structure to store graph information"""
    adjacency_matrix: np.ndarray
    edge_probabilities: np.ndarray
    edge_features: np.ndarray
    node_features: np.ndarray
    edge_list: List[Tuple[int, int]]
    start: np.ndarray
    goal: np.ndarray
    action: np.ndarray
    value: np.ndarray
    graph_metadata: Dict


def convert_vertex_2vector(vertex: int) -> np.ndarray:
    """Convert vertex indices to 2D vectors"""
    return np.array([[vertex]])

def convert_value_2vector(value: float) -> np.ndarray:
    """Convert scalar value to 1D vector"""
    return np.array([[value]])


def create_graph_datum(
    graph: ug.ProbabilisticGraph,
    action: List[int],
    start: int,
    goal: int,
    value: float,
    metadata: Optional[Dict] = None
) -> GraphData:
    
    # create_graph_data(graph=pg, action=edge, start=start, goal=goal, value=bc)
    """
    Create a complete GraphData object
    
    Args:
        graph: NetworkX graph object
        edge_probs: Dictionary of edge probabilities
        node_feature_types: Types of node features to compute
        edge_feature_types: Types of edge features to compute
        metadata: Additional metadata to store
    
    Returns:
        GraphData object containing all graph information
    """
    # Create adjacency matrix
    adjacency_matrix = graph.adjacency
    # create edge probability matrix
    edge_prob_matrix = graph.probabilities
    # start and goal as node features
    start_vector = convert_vertex_2vector(start)
    goal_vector = convert_vertex_2vector(goal)
    value_vector = convert_value_2vector(value)
    action_convert = [convert_vertex_2vector(action[0]), convert_value_2vector(action[1])]
    action_vector = np.array(action_convert).reshape(1, -1)
    
    
    # Create edge list
    nxgraph = nx.Graph(adjacency_matrix)
    edge_list = list(nxgraph.edges())
    
    # Create metadata
    if metadata is None:
        metadata = {}
    metadata.update({
        'num_nodes': adjacency_matrix.shape[0],
        'num_edges': np.count_nonzero(adjacency_matrix) // 2,
        'graph_density': nx.density(nxgraph),
        'is_connected': nx.is_connected(nxgraph),
    })
    
    return GraphData(
        adjacency_matrix=adjacency_matrix.astype(np.float32),
        edge_probabilities=edge_prob_matrix.astype(np.float32),
        edge_list=edge_list,
        start=start_vector.astype(np.float32),
        goal=goal_vector.astype(np.float32),
        action=action_vector.astype(np.float32),
        value=value_vector.astype(np.float32),
        num_nodes=nxgraph.number_of_nodes(),
        num_edges=nxgraph.number_of_edges(),
        graph_metadata=metadata,
    )


def generate_dataset(
    path: str,
    num_graphs: int,
    num_maps: int = 500,
) -> List[GraphData]:
    """
    Generate a dataset of multiple graphs
    Args:
        num_graphs: Number of graphs to generate
        graph_type: Type of graph ('random', 'scale_free', 'small_world')
        num_nodes_range: Range of number of nodes (min, max)
        graph_params: Parameters for graph generation
        prob_params: Parameters for probability distribution
        node_feature_types: Types of node features
        edge_feature_types: Types of edge features
        seed: Random seed for reproducibility
    
    Returns:
        List of GraphData objects
    """
    dataset = []
    num_data_per_graph = 100
    seeds = random.sample(range(1000, 10000), num_graphs)
    for i in range(num_graphs):
        np.random.seed(seeds[i])
        random.seed(seeds[i])
        # Generate random number of nodes
        _, _, graph = graphs.get_insland_bridges_graph()
        edges = ug.get_initial_edges(graph)
        vertex_positions = ug.get_vertex_positions(graph.vertices)
        adjacency_matrix, probability_matrix = ug.create_adj_prob_matrices(edges, vertex_positions)
            
        for poi in graph.pois:
            edge = [poi.neighbors[0]-1, poi.neighbors[1]-1] # calculate its value
            count = 0
            while count < num_data_per_graph:
                start, goal = random.sample(range(0, len(graph.vertices)), 2)
                num_known_edges = random.randint(0, len(graph.pois)-1)
                known_edges = random.sample(graph.pois, num_known_edges)
                known_edge_list = [0.0 if random.random() >= poi.block_prob else 1.0 for poi in known_edges]
                known_edges_id = [[poi.neighbors[0]-1, poi.neighbors[1]-1] for poi in known_edges]
                if edge in known_edges_id:
                    known_edges_id.remove(edge)
                new_probability_matrix = ug.set_edge_probabilities(
                    probs=np.array(known_edge_list),
                    edges=known_edges_id,
                    probabilities=probability_matrix
                )
                pg = ug.ProbabilisticGraph(
                    positions=vertex_positions,
                    adjacency=adjacency_matrix,
                    probabilities=new_probability_matrix
                )
                bc = ae.get_single_bc_networkX(
                    ugraph=pg,
                    action_edge=[edge],
                    start=start,
                    goalID=goal,
                    n_samples=num_maps
                )
                graph_data = create_graph_datum(graph=pg, action=edge, start=start, goal=goal, value=bc)
                write_datum_to_file(path, seeds[i], graph_data, count)
                # dataset.append(graph_data)
                count += 1
    # return dataset

def write_datum_to_file(path, seed, datum, counter):
    """Write a single datum to file and append name to csv record."""
    # Get the data file name
    data_filename = os.path.join('pickles', f'dat_{seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(
        os.path.join(path, data_filename), datum)
    csv_filename = f'{path}_{seed}.csv'
    with open(os.path.join(path, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')
        

def _setup(args):
    # print(f"Planner: {args.planner}, a team of {args.num_ugvs}UGV(s)-{args.num_drones}UAV(s), iters.: {args.num_iterations},"
    #       f" max depth: {args.max_depth}, maps: {args.sampling_maps}, AVP:{use_AVP}, DAP:{use_DAP}") 
    
    generate_dataset(path=args.save_dir, num_graphs= args.num_graphs,
                        num_maps = args.num_maps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--num_maps', type=int, default=500)
    parser.add_argument('--num_graphs', type=int, default=5)
    args = parser.parse_args()

    _setup(args)
