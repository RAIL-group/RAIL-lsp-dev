import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool

NODE_IN = 2
EDGE_IN = 2
HIDDEN = 64

class BipartiteEdgeRegressor(nn.Module):
    def __init__(self, node_in_dim=2, edge_in_dim=2, hidden_dim=32, num_heads=4):
        super(BipartiteEdgeRegressor, self).__init__()

        # --- 1. Projections ---
        # Project distinct features to the same hidden dimension
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

        # --- 2. Bipartite GAT Layers ---
        # Layer 1: Nodes -> Edges
        # Input: (Node_Features, Edge_Features)
        # Output: Updated Edge_Features
        self.gat_nodes_to_edges = GATv2Conv(
            in_channels=(hidden_dim, hidden_dim), # (Source dim, Target dim)
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False, # Average the heads to keep dim constant
            add_self_loops=False # Bipartite graphs can't have self-loops
        )

        # Layer 2: Edges -> Nodes
        # Input: (Edge_Features, Node_Features)
        # Output: Updated Node_Features
        self.gat_edges_to_nodes = GATv2Conv(
            in_channels=(hidden_dim, hidden_dim),
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,
            add_self_loops=False
        )

        # Layer 3: Nodes -> Edges (Final Refinement)
        self.gat_final = GATv2Conv(
            in_channels=(hidden_dim, hidden_dim),
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,
            add_self_loops=False
        )

        # --- 3. Regression Head ---
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # Force positive output
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: [Num_Nodes, Node_Feats] (Start/Goal)
        edge_index: [2, Num_Edges] (Connectivity)
        edge_attr: [Num_Edges, Edge_Feats] (Length/Prob)
        """

        # --- Preprocessing: Create Bipartite Connectivity ---
        # We need to define which Nodes connect to which Edges.
        # Original edge_index is [2, M].
        # Row 0 is Source Nodes, Row 1 is Target Nodes.
        # We have M edges, indexed 0 to M-1.

        num_edges = edge_index.size(1)
        device = x.device

        # Create indices for the "Edge Nodes"
        edge_indices = torch.arange(num_edges, device=device)

        # Connection Set 1: Source Nodes -> Edge Nodes
        # Connection Set 2: Target Nodes -> Edge Nodes
        # We combine them because the road is bidirectional (undirected).
        # Node u connects to Edge e, Node v connects to Edge e.

        # Source indices (Nodes)
        node_idx_all = edge_index.t().contiguous().view(-1) # [Source_0, Target_0, Source_1, Target_1...]

        # Target indices (Edges)
        # We repeat each edge index twice: [Edge_0, Edge_0, Edge_1, Edge_1...]
        edge_idx_all = edge_indices.repeat_interleave(2)

        # This is the connectivity matrix for Nodes -> Edges
        # Shape: [2, 2*M]
        bipartite_index = torch.stack([node_idx_all, edge_idx_all], dim=0)

        # For Edges -> Nodes, we just flip this index
        bipartite_index_transpose = torch.stack([edge_idx_all, node_idx_all], dim=0)

        # --- Forward Pass ---

        # 1. Initial Projection
        h_nodes = F.relu(self.node_proj(x))         # [N, Hidden]
        h_edges = F.relu(self.edge_proj(edge_attr)) # [M, Hidden]

        # 2. Layer 1: Nodes pass info to Edges
        # "Edges look at their endpoints"
        # Input tuple: (Source, Target) -> (Nodes, Edges)
        h_edges = self.gat_nodes_to_edges((h_nodes, h_edges), bipartite_index)
        h_edges = F.relu(h_edges)

        # 3. Layer 2: Edges pass info back to Nodes
        # "Nodes look at connected roads"
        # Input tuple: (Source, Target) -> (Edges, Nodes)
        # Note: We use the transposed index here
        h_nodes = self.gat_edges_to_nodes((h_edges, h_nodes), bipartite_index_transpose)
        h_nodes = F.relu(h_nodes)

        # 4. Layer 3: Nodes pass info to Edges again (Final Context)
        h_edges = self.gat_final((h_nodes, h_edges), bipartite_index)
        h_edges = F.relu(h_edges)
        blocking_prob = edge_attr[:, 1]  # Extract the blocking probability from edge attributes
        mask = (blocking_prob > 0.0) & (blocking_prob < 1.0)  # Only consider edges that are not deterministic        
        return self.regressor(h_edges), mask

    
    def loss(self, preds, targets, masks):
        # assert 1 == 0
        # MSE Loss for regression
        masked_preds = preds[masks]
        masked_targets = targets[masks]
        # assert  targets[masks] >= 0.0, "Targets must be non-negative for"
        if masked_preds.numel() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        return F.mse_loss(masked_preds, masked_targets)


    def ig_regression_loss(self,
        pred:      Tensor,   # [E]
        target:    Tensor,   # [E]  from prepare_ig_labels()
        edge_attr: Tensor,   # [E, 2]
        uncertain_weight: float = 1.0,
        certain_weight:   float = 0.1,
    ) -> Tensor:
        """
        Weighted Huber loss.
        Uncertain edges: full weight  (primary learning signal).
        Certain  edges : small weight (boundary regularisation only).
        """
        p         = edge_attr[:, 1]
        uncertain = (p > 0.0) & (p < 1.0)
        certain   = ~uncertain

        weights            = torch.zeros_like(pred)
        weights[uncertain] = uncertain_weight
        weights[certain]   = certain_weight

        element_loss = F.huber_loss(pred, target, reduction='none')  # [E]
        return (element_loss * weights).sum() / weights.sum()


def load_iap_gnn_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BipartiteEdgeRegressor(node_in_dim=NODE_IN, edge_in_dim=EDGE_IN, hidden_dim=HIDDEN).to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model