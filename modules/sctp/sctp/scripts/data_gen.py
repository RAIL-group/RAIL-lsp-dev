import os
import random, argparse
import numpy as np

import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import random
import sctp.sctp_graphs as graphs
import sctp.utils.underlying_graph as ug
import sctp.action_estimation as ae

import learning


@dataclass
class GraphData:
    """Data structure to store graph information"""
    x: np.ndarray #start, goal
    edge_index: List #edge_index
    edge_attr: np.ndarray #edge_attr
    y: np.ndarray #y [M ,1] M =edges_num valuees - y
    graph_metadata: Dict


def create_graph_datum(
    graph: ug.ProbabilisticGraph,
    edges: List,
    start: int,
    goal: int,
    values: np.ndarray,
    metadata: Optional[Dict] = None
) -> GraphData:
    # Create adjacency matrix
    nodes = [[0,0] for _ in range(graph.adjacency.shape[0])]
    assert len(nodes) == 16
    nodes[start] = [1,0]
    nodes[goal] = [0,1]
    edge_attr = [[graph.adjacency[edge[0], edge[1]], graph.probabilities[edge[0], edge[1]]] for edge in edges]
    
    # Create metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'num_nodes': graph.adjacency.shape[0],
        'num_edges': np.count_nonzero(graph.adjacency) // 2,
    })
    
    return GraphData(
        x=np.array(nodes, dtype=np.int64),
        edge_index=np.array(edges, dtype=np.int64),
        edge_attr=np.array(edge_attr, dtype=np.float32),
        y=np.array(values, dtype=np.float32).reshape(-1, 1),
        graph_metadata=metadata,
    )


def generate_dataset(
    filepath: str,
    num_graphs: int,
    num_maps: int = 500,
    graph_type: str = 'bridges',
    num_data_per_graph: int = 50,
):
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
    seeds = np.arange(1000, 1000+num_graphs)
    for i in range(num_graphs):
        np.random.seed(seeds[i])
        random.seed(seeds[i])
        # Generate random number of nodes
        if graph_type == 'island':
            _, _, graph = graphs.get_sixIslands_graph()
        elif graph_type == 'random':
            _, _, graph = graphs.random_graph()
        elif graph_type == 'bridges':
            _, _, graph = graphs.get_bridges_graph()
        else:
            raise ValueError(f"Graph type {graph_type} not recognized")
        edges = ug.get_initial_edges(graph)
        vertex_positions = ug.get_vertex_positions(graph.vertices)
        adjacency_matrix, probability_matrix = ug.create_adj_prob_matrices(edges, vertex_positions)
        count = 0
        while count < num_data_per_graph:
            start, goal = random.sample(range(0, len(graph.vertices)), 2)
            num_known_edges = random.randint(0, len(graph.pois)-1)
            known_edges = random.sample(graph.pois, num_known_edges)
            known_edge_probs = [0.0 if random.random() >= poi.block_prob else 1.0 for poi in known_edges]
            known_edges_id = [[poi.neighbors[0]-1, poi.neighbors[1]-1] for poi in known_edges]
            new_probability_matrix = ug.set_edge_probabilities(
                probs=np.array(known_edge_probs),
                edges=known_edges_id,
                probabilities=probability_matrix
            )
            pg = ug.ProbabilisticGraph(
                positions=vertex_positions,
                adjacency=adjacency_matrix,
                probabilities=new_probability_matrix
            )
            values = []
            edge_list = []
            for poi in graph.pois:
                edge = [poi.neighbors[0]-1, poi.neighbors[1]-1] # calculate its value
                assert edge[0] < edge[1], f"Edge {edge} is not in the correct order"
                edge_list.append(edge)
                if edge in known_edges_id:
                    values.append(0.0)
                    continue    
                
                bc = ae.get_single_bc_networkX(
                    ugraph=pg,
                    action_edge=[edge],
                    start=start,
                    goalID=goal,
                    n_samples=num_maps
                )
                values.append(bc)
            assert len(edge_list) == len(values), f"Number of edges {len(edge_list)} does not match number of values {len(values)}"
                
            graph_data = create_graph_datum(graph=pg, edges=edge_list, start=start, goal=goal, values=values)
            write_datum_to_file(filepath, seeds[i], graph_data, count)
            count += 1

def write_datum_to_file(filepath, seed, datum, counter):
    """Write a single datum to file and append name to csv record."""
    # Get the data file name
    data_filename = os.path.join('pickles', f'dat_{seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(os.path.join(filepath, data_filename), datum)
    csv_filename = f'graph_data_address.csv'
    with open(os.path.join(filepath, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')
        

def _setup(args):
    print(f"Graph_Type: {args.graph_type}, number of maps for sampling {args.num_maps}-number of graph: {args.num_graphs},"
          f" saving to: {args.save_dir}") 
    
    generate_dataset(filepath=args.save_dir, num_graphs= args.num_graphs,
                        num_maps = args.num_maps, graph_type=args.graph_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--num_maps', type=int, default=500)
    parser.add_argument('--num_graphs', type=int, default=5)
    parser.add_argument('--graph_type', type=str, default='bridges')
    args = parser.parse_args()

    _setup(args)
