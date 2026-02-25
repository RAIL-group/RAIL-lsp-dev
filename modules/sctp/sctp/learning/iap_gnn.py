import torch
import torch.nn as nn
import numpy as np
# from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class BipartiteEdgeRegressor(nn.Module):
    def __init__(self, node_in_dim=2, edge_in_dim=2, hidden_dim=32, num_heads=2):
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
        node_idx_all = edge_index.flatten() # [Source_0, Target_0, Source_1, Target_1...]

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

        # 5. Predict on Edges
        return self.regressor(h_edges)

    
    def loss(self, pred, target, mask=None):
        # MSE Loss for regression
        return F.mse_loss(pred.view(-1), target.view(-1))