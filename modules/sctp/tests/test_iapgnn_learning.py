import numpy as np
import random, os
import glob
from sctp.learning.iap_gnn import BipartiteEdgeRegressor
from sctp.learning.iap_gnn2 import EdgeIGGNN
from sctp.learning import iap_gnn2
from sctp.scripts.data_gen import generate_dataset
from sctp.scripts.iapgnn_training import GzipGNNDataset, train_epoch
import pickle, gzip
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.optim as optim

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

def test_get_Graphdata():
    # Load the dataset
    with gzip.open('data/sctp/graph_data/pickles/dat_1000_1.pgz', 'rb') as f:
        data = pickle.load(f)
    print("Node Features (x):", data.x.shape)  # [Num_Nodes, Node_Feats]
    print("Node Features (x):", data.x)  # [Num_Nodes, Node_Feats]
    print("Edge Index:", data.edge_index.shape)  # [2, Num_Edges]
    print("Edge Index:", data.edge_index)  # [2, Num_Edges]
    print("Edge Attributes:", data.edge_attr.shape)  # [Num_Edges, Edge_Feats]
    print("Edge Attributes:", data.edge_attr)  # [Num_Edges, Edge_Feats]
    print("Target Values (y):", data.y.shape)  # [Num_Edges, 1]
    print("Target Values (y):", data.y)  # [Num_Edges, 1]

def test_IAPtraining():
    NODE_IN = 2
    EDGE_IN = 2
    HIDDEN = 126
    
    # Check the first data point
    all_files = glob.glob(os.path.join('data/sctp/graph_data/pickles/', '*.pgz'))
    dataset = GzipGNNDataset(all_files)
    # Use a DataLoader for batching
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BipartiteEdgeRegressor(NODE_IN, EDGE_IN, HIDDEN).to(device)
    # device = torch.device('cpu')  # Force CPU for debugging
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    print("Starting training...")
    num_epochs = 4000
    for epoch in range(1, num_epochs+1):
        loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")
        
    
def test_IAP_newGNN():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── example graph ──────────────────────────────────────────────────────
    n_nodes = 5
    x = torch.tensor([
        [1., 0.],  # 0 start
        [0., 0.],  # 1
        [0., 0.],  # 2
        [0., 0.],  # 3
        [0., 1.],  # 4 goal
    ])

    # Forward edges (u < v by convention)
    fwd_edges = [          # (u, v, distance, p_blocking)
        (0, 1, 1.0, 0.3),  # uncertain
        (1, 2, 2.0, 0.0),  # certain: always passable  → IG = 0
        (2, 4, 1.5, 0.5),  # uncertain
        (0, 3, 3.0, 0.7),  # uncertain
        (3, 4, 2.5, 1.0),  # certain: always blocked   → IG = 0
    ]
    E = len(fwd_edges)
    src_f, dst_f, dists, probs = zip(*fwd_edges)

    # Forward-only edge_index and edge_attr — reverse is built inside forward()
    edge_index = torch.tensor([list(src_f), list(dst_f)], dtype=torch.long)
    edge_attr  = torch.tensor(list(zip(dists, probs)), dtype=torch.float)

    # ── assert structural correctness ──────────────────────────────────────
    print("Checking edge_index/edge_attr consistency...")
    iap_gnn2.assert_edge_symmetry(edge_index, edge_attr)
    print("  ✓ edge_index is [2, E] and edge_attr is [E, edge_dim]\n")

    # ── ground-truth IG labels ([E] values, one per forward edge) ──────────
    ig_raw = torch.tensor([0.45, -0.01, 0.30, 0.80, 0.02])
    #                       ^unc   ^cert  ^unc  ^unc  ^cert

    print(f"Raw IG labels   : {ig_raw.tolist()}")
    ig_clean = iap_gnn2.prepare_ig_labels(ig_raw, edge_attr)
    print(f"Cleaned labels  : {ig_clean.tolist()}")
    print("  ✓ negatives clamped, certain edges zeroed\n")

    # ── assert certain edges are zero ──────────────────────────────────────
    print("Checking certain-edge constraint on cleaned labels...")
    iap_gnn2.assert_certain_edges_zero(ig_clean, edge_attr)
    print("  ✓ all certain edges have IG = 0.0\n")

    # ── deliberate failure demo ────────────────────────────────────────────
    print("Testing assert with bad labels (p=1 edge given IG=0.5)...")
    bad_labels = ig_clean.clone()
    bad_labels[4] = 0.5   # edge (3,4) has p=1 → should be 0
    try:
        iap_gnn2.assert_certain_edges_zero(bad_labels, edge_attr)
    except AssertionError as e:
        print(f"  ✓ AssertionError caught correctly:\n    {e}\n")

    # ── model forward pass ─────────────────────────────────────────────────
    # data = Data(
    #     x          = x,
    #     edge_index = edge_index,
    #     edge_attr  = edge_attr,
    #     ig_labels  = ig_clean,
    # )

    # model = EdgeIGGNN(
    #     node_dim=2, edge_dim=2, hidden_dim=64, n_heads=4, n_layers=3, dropout=0.1
    # ).to(device)

    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Trainable parameters: {n_params:,}")
    # print(f"Output shape expected: [{E}]\n")

    # with torch.no_grad():
    #     ig_out = model(
    #         x          = data.x.to(device),
    #         edge_index = data.edge_index.to(device),
    #         edge_attr  = data.edge_attr.to(device),
    #     )

    # print(f"Output shape : {ig_out.shape}  ← should be [{E}]")
    # print(f"Output values: {[round(v, 4) for v in ig_out.tolist()]}")
    # print(f"\nCertain edge outputs (should be 0.0):")
    # print(f"  edge (1,2) p=0.0 → IG = {ig_out[1].item():.6f}")
    # print(f"  edge (3,4) p=1.0 → IG = {ig_out[4].item():.6f}")

    # # ── retrieval by edge ──────────────────────────────────────────────────
    # pos = iap_gnn2.get_edge_position(data.edge_index, 0, 3)
    # print(f"\nEdge (0,3) is at index {pos},  IG = {ig_out[pos].item():.4f}")

    # # ── all uncertain edges ────────────────────────────────────────────────
    # uncertain_igs = iap_gnn2.get_all_uncertain_ig(ig_out.cpu(), data.edge_index, data.edge_attr)
    # print("\nUncertain edge IG values (untrained — inspection only):")
    # for r in uncertain_igs:
    #     print(f"  [{r['edge_idx']}] edge {r['edge']}  p={r['p_block']:.1f}  IG={r['ig_score']:.4f}")

    # print("\nSmoke-test passed ✓")