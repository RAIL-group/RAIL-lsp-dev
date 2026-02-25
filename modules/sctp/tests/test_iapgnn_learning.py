import numpy as np
import random
from sctp.learning.iap_gnn import BipartiteEdgeRegressor
from sctp.scripts.data_gen import generate_dataset
import pickle, gzip
from torch_geometric.data import Data
import torch

def test_data_generation():
    graph_type = 'bridges'
    sampling_nums = 500
    graph_nums = 10
    num_data_per_graph = 50
    filepath ='data/sctp/graph_data/'
    print(f"Graph_Type: {graph_type}, number of maps for sampling {sampling_nums}-number of graph: {graph_nums},"
          f" saving to: {filepath}") 
    
    generate_dataset(filepath=filepath, num_graphs=graph_nums, num_maps=sampling_nums, 
                        graph_type=graph_type, num_data_per_graph=num_data_per_graph)


def test_GNN_model():
    # Hyperparameters
    NODE_IN = 2  # [Start, Goal]
    EDGE_IN = 2  # [Length, Prob] <-- Fixed: Changed from 3 to 2 to match data
    HIDDEN = 32

    model = BipartiteEdgeRegressor(NODE_IN, EDGE_IN, HIDDEN)

    # Dummy Data
    x = torch.tensor([[1,0], [0,0], [0,1], [0,0]], dtype=torch.float) # 4 Nodes
    # hot code to define start node and goal.
    
    # how about the Node we need to calculate the value?

    # Fixed: Transpose edge_index to [2, Num_Edges]
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long).t().contiguous()
    # all connectivity between nodes (edges)
    
    edge_attr = torch.tensor([[5.0, 0.1], [4.0, 0.0], [4.0, 0.5]], dtype=torch.float)
    # [distance, prob]
    
    # Forward
    preds = model(x, edge_index, edge_attr)

    print(preds.shape) # Output: [3, 1] (One prediction per edge)
    print(preds)

def test_GNN_pickle_data():
    # Load the dataset
    with gzip.open('data/sctp/graph_data/pickles/dat_1000_0.pgz', 'rb') as f:
        dataset = pickle.load(f)

    # Check the first data point
    data = dataset
    print("Node Features (x):", data.x.shape)  # [Num_Nodes, Node_Feats]
    print("Edge Index:", data.edge_index.shape)  # [2, Num_Edges]
    print("Edge Attributes:", data.edge_attr.shape)  # [Num_Edges, Edge_Feats]
    print("Target Values (y):", data.y.shape)  # [Num_Edges, 1]
    NODE_IN = 2
    EDGE_IN = 2
    HIDDEN = 32
    model = BipartiteEdgeRegressor(NODE_IN, EDGE_IN, HIDDEN)
    x = torch.tensor(data.x, dtype=torch.float)
    edge_index = torch.tensor(data.edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(data.edge_attr, dtype=torch.float)
    
    preds = model(x, edge_index, edge_attr)
    print(preds)
    print(preds.shape)
    
    