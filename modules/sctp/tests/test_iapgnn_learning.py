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
from sklearn.model_selection import train_test_split

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
        
    
def test_IAPGNN_predictions():
    print()
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    NODE_IN = 2
    EDGE_IN = 2
    HIDDEN = 64
    model = BipartiteEdgeRegressor(node_in_dim=NODE_IN, edge_in_dim=EDGE_IN, hidden_dim=HIDDEN).to(device)
    

    PATH = "/modules/sctp/learning/models/iap_gnn.pt" # The path to your saved model file

    # 3. Load the state dictionary
    # Use weights_only=True as a best practice
    model.load_state_dict(torch.load(PATH, weights_only=True))
    
    # test_model(model, dataset, device, n_samples=2)
    """
    Run inference on a random subset of graphs and print predicted vs
    ground-truth IG values side-by-side for every edge.

    Parameters
    ----------
    model      : trained EdgeIGGNN
    dataset    : list of PyG Data objects (each must have ig_labels)
    device     : torch device
    n_samples  : number of graphs to sample for inspection
    seed       : random seed for reproducible sampling
    """
    model.eval()
    # torch.manual_seed(seed)
    data_dir = 'data/sctp/graph_data/pickles/'
    all_files = glob.glob(os.path.join(data_dir, '*.pgz'))
    n_samples = 1
    dataset = GzipGNNDataset(all_files)

    indices = torch.randperm(len(dataset))[:n_samples].tolist()

    # ── aggregate metrics across all sampled graphs ────────────────────────
    all_pred    = []
    all_target  = []

    print("=" * 75)
    print(f"  MODEL TEST RESULTS  ({n_samples} randomly sampled graphs)")
    print("=" * 75)

    for sample_num, idx in enumerate(indices, 1):
        data   = dataset[idx].to(device)
        # target = prepare_ig_labels(data.ig_labels, data.edge_attr)

        with torch.no_grad():
            pred, _ = model(
                x          = data.x,
                edge_index = data.edge_index,
                edge_attr  = data.edge_attr,
            )   # [E]

        pred_cpu   = pred.cpu()
        target_cpu = data.y.cpu()
        E          = pred_cpu.shape[0]

        src = data.edge_index[0].cpu()   # [E]
        dst = data.edge_index[1].cpu()   # [E]
        p   = data.edge_attr[:, 1].cpu() # [E]

        uncertain_mask = (p > 0.0) & (p < 1.0)
        certain_mask   = ~uncertain_mask

        # ── per-graph header ───────────────────────────────────────────────
        print(f"\n{'─' * 75}")
        print(f"  Graph {sample_num}  (dataset index {idx})  |  "
              f"{E} edges  |  "
              f"{uncertain_mask.sum().item()} uncertain  |  "
              f"{certain_mask.sum().item()} certain")
        print(f"{'─' * 75}")
        print(f" {'Edge':>7}  {'p_block':>10}  {'Type':>8}  "
              f"{'GT-IG':>10}  {'Pred IG':>10}  {'Error':>10}  {'AbsErr':>9}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

        for i in range(E):
            u      = src[i].item()
            v      = dst[i].item()
            p_val  = p[i].item()
            gt     = target_cpu[i].item()
            pr     = pred_cpu[i].item()
            err    = pr - gt
            abs_err= abs(err)
            etype  = "uncertain" if uncertain_mask[i] else "certain"

            # flag large errors
            flag = " ◄" if abs_err > 1.0 and uncertain_mask[i] else ""

            print(f"  ({u:>2},{v:>2})     "
                  f"{p_val:>5.3f}  "
                  f"{etype:>10}  "
                  f"{gt:>9.4f}  "
                  f"{pr:>10.4f}  "
                  f"{err:>+11.4f}  "
                  f"{abs_err:>8.4f}"
                  f"{flag}")

            if uncertain_mask[i]:
                all_pred.append(pr)
                all_target.append(gt)

        # ── per-graph summary ──────────────────────────────────────────────
        unc_pred   = pred_cpu[uncertain_mask]
        unc_target = target_cpu[uncertain_mask]

        if uncertain_mask.any():
            mae  = (unc_pred - unc_target).abs().mean().item()
            rmse = ((unc_pred - unc_target) ** 2).mean().sqrt().item()
            bias = (unc_pred - unc_target).mean().item()

            # ranking accuracy: fraction of pairs ordered correctly
            dp = unc_pred.unsqueeze(0)   - unc_pred.unsqueeze(1)    # [K,K]
            dt = unc_target.unsqueeze(0) - unc_target.unsqueeze(1)  # [K,K]
            pairs = dt.abs() > 0.01
            rank_acc = ((dp * dt) > 0)[pairs].float().mean().item() if pairs.any() else float('nan')

            print(f"\n  Graph {sample_num} uncertain-edge metrics:")
            print(f"    MAE        = {mae:.4f}")
            print(f"    RMSE       = {rmse:.4f}")
            print(f"    Bias       = {bias:+.4f}  "
                  f"({'over-predicting' if bias > 0 else 'under-predicting'})")
            print(f"    Rank Acc   = {rank_acc:.3f}  "
                  f"(fraction of edge pairs ranked correctly)")
        else:
            print(f"\n  Graph {sample_num}: no uncertain edges to evaluate.")

    # ── global summary across all sampled graphs ───────────────────────────
    # if all_pred:
    #     all_pred   = torch.tensor(all_pred)
    #     all_target = torch.tensor(all_target)

    #     mae  = (all_pred - all_target).abs().mean().item()
    #     rmse = ((all_pred - all_target) ** 2).mean().sqrt().item()
    #     bias = (all_pred - all_target).mean().item()
    #     rel_err = ((all_pred - all_target).abs() /
    #                (all_target.abs() + 1e-8)).mean().item() * 100

    #     dp = all_pred.unsqueeze(0)   - all_pred.unsqueeze(1)
    #     dt = all_target.unsqueeze(0) - all_target.unsqueeze(1)
    #     pairs    = dt.abs() > 0.01
    #     rank_acc = ((dp * dt) > 0)[pairs].float().mean().item() if pairs.any() else float('nan')

    #     print(f"\n{'=' * 75}")
    #     print(f"  GLOBAL SUMMARY  (uncertain edges only, {len(all_pred)} total)")
    #     print(f"{'=' * 75}")
    #     print(f"  MAE            = {mae:.4f}")
    #     print(f"  RMSE           = {rmse:.4f}")
    #     print(f"  Bias           = {bias:+.4f}  "
    #           f"({'over-predicting' if bias > 0 else 'under-predicting'})")
    #     print(f"  Rel. Error     = {rel_err:.2f}%")
    #     print(f"  Rank Accuracy  = {rank_acc:.3f}")
    #     print(f"{'=' * 75}\n")


    