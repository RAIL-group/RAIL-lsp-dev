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
    edge_index: np.ndarray #edge_index
    edge_attr: np.ndarray #edge_attr
    y: np.ndarray #y [M ,1]
    graph_metadata: Dict


def create_graph_datum(
    graph: ug.ProbabilisticGraph,
    edges: List,
    start: int,
    goal: int,
    values: np.ndarray,
    metadata: Optional[Dict] = None
) -> GraphData:
    # 1. Create node features
    # 1. Create node features [Num_Nodes, 2]
    num_nodes = graph.adjacency.shape[0]
    nodes = np.zeros((num_nodes, 2), dtype=np.float32)
    # Mark start and goal using one-hot style features
    nodes[start, 0] = 1.0
    nodes[goal, 1] = 1.0
    
    edge_attr = []
    for u, v in edges:
        length = graph.adjacency[u, v]
        prob = graph.probabilities[u, v]
        edge_attr.append([length, prob])
    
    # Create metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'num_nodes': num_nodes,
        'num_edges': len(edges),
    })
    
    return GraphData(
        x= nodes,
        edge_index=np.array(edges, dtype=np.int64).T,
        edge_attr=np.array(edge_attr, dtype=np.float32),
        y=np.array(values, dtype=np.float32).reshape(-1, 1),
        graph_metadata=metadata,
    )


def generate_dataset(
    filepath: str,
    seed: int,
    num_maps: int = 500,
    graph_type: str = 'bridges',
    num_data_per_graph: int = 50,
    verbose=False
):
    """
    Generate a dataset of multiple graphs
    Args:
        num_graphs: Number of graphs to generate
        graph_type: Type of graph ('random', 'bridges', 'islands')
    
    Returns:
        List of GraphData objects
    """
    assert num_maps == 1000
    # seeds = np.arange(1021, 1021+num_graphs)
    # for i in range(num_graphs):
    np.random.seed(seed)
    random.seed(seed)
    # Generate random number of nodes
    if graph_type == 'islands':
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
    if verbose:
        file_summary = os.path.join('pickles_new', f'dat_{graph_type}_{seed}.txt')
        file_summary = os.path.join(filepath, file_summary)
    
    vertices_id = [vertex.id for vertex in graph.vertices]
    for ii in range(len(vertices_id)-1):
        assert vertices_id[ii] < vertices_id[ii+1], f"Vertices are not in the correct order: {vertices_id}"
        
    while count < num_data_per_graph:
        start, goal = random.sample(range(0, len(graph.vertices)), 2)
        num_known_edges = random.randint(0, 10)
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
            values.append(bc) if bc >= 0.0 else values.append(0.0)
        assert len(edge_list) == len(values), f"Number of edges {len(edge_list)} does not match number of values {len(values)}"
        # min_val = min(values)
        # make all values no negative but keep the 0.0 values as they are (indicating known edges)
        # if min_val < 0.0:
        #     values = [v-min_val if v != 0.0 else 0.0 for v in values]            
        graph_data = create_graph_datum(graph=pg, edges=edge_list, start=start, goal=goal, values=values)
        write_datum_to_file(filepath=filepath, seed=seed, datum=graph_data, counter=count, graph_type=graph_type)
        count += 1
        if verbose:
            with open(file_summary, "a+") as f:
                f.write(f"START: {start} | GOAL: {goal}\n")
                f.write(f"EDGE_LIST: {edge_list}\n")
                f.write(f"VALUES: {values}\n")
            


def write_datum_to_file(filepath, seed, datum, counter, graph_type='bridges'):
    """Write a single datum to file and append name to csv record."""
    # Get the data file name
    data_filename = os.path.join('pickles_new', f'dat_{graph_type}_{seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(os.path.join(filepath, data_filename), datum)
    csv_filename = f'graph_data_address_new.csv'
    with open(os.path.join(filepath, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')
        

def _setup(args):
    print(f"Graph_Type: {args.graph_type}, number of maps: {args.num_maps}-with seed: {args.seed},"
          f" saving to: {args.save_dir}") 
    verbose = False
    generate_dataset(filepath=args.save_dir, seed=args.seed, num_maps=args.num_maps, \
                        graph_type=args.graph_type, verbose=verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--num_maps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--graph_type', type=str, default='bridges')
    args = parser.parse_args()

    _setup(args)
