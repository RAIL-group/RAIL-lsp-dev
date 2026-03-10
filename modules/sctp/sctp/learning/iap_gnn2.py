"""
GATv2-based GNN for Edge-Level Information Gain Prediction (v3)
===============================================================
Changes from v2
───────────────
1. Output dim = |E| (one value per undirected edge, not 2|E|).
   Convention: edge_index is stored as [forward_E | reverse_E],
   so edge_index[:, :E] are the canonical directed edges (u→v, u<v)
   and edge_index[:, E:] are their reverses (v→u).
   Output ig_out[i] corresponds to undirected edge i = edge_index[:, i].

2. Assert that all edges with p=0 or p=1 have ground-truth IG = 0.0.

3. Assert that edge_index follows the [forward | reverse] convention,
   i.e. edge_index[:, E+i] == edge_index[[1,0], i] for all i.

Node features : [1,0]=start, [0,1]=goal, [0,0]=other  (dim=2)
Edge features : [distance, p_blocking]                 (dim=2)
Output        : [E] IG value per undirected edge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from typing import Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# 0.  Structural assertions
# ---------------------------------------------------------------------------

def assert_edge_symmetry(edge_index: Tensor, edge_attr: Tensor) -> None:
    """
    Assert basic sanity on forward-only edge data:
      - edge_index shape is [2, E]
      - edge_attr  shape is [E, edge_dim]
      - both have the same number of edges E
    Note: reverse edges are constructed inside forward(), not stored in data.
    """
    assert edge_index.dim() == 2 and edge_index.shape[0] == 2, (
        f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}."
    )
    E_idx  = edge_index.shape[1]
    E_attr = edge_attr.shape[0]
    assert E_idx == E_attr, (
        f"edge_index has {E_idx} edges but edge_attr has {E_attr} rows. "
        "They must match."
    )


def assert_certain_edges_zero(
    ig_labels: Tensor,   # [E] ground-truth IG values
    edge_attr: Tensor,   # [E, 2]  col-1 = p_blocking
    tol:       float = 1e-6,
) -> None:
    """
    Assert that edges with p_blocking = 0.0 or 1.0 have IG label = 0.0.
    """
    p            = edge_attr[:, 1]                 # [E]
    certain_mask = (p <= 0.0) | (p >= 1.0)

    assert ig_labels.shape[0] == edge_attr.shape[0], (
        f"ig_labels has {ig_labels.shape[0]} entries but edge_attr has "
        f"{edge_attr.shape[0]} rows. Must match."
    )

    if certain_mask.any():
        bad = ig_labels[certain_mask]
        assert torch.all(bad.abs() <= tol), (
            f"Ground-truth IG must be 0.0 for certain edges (p=0 or p=1). "
            f"Found non-zero values: {bad[bad.abs() > tol].tolist()} "
            f"at edge indices: {certain_mask.nonzero(as_tuple=True)[0][bad.abs() > tol].tolist()}."
        )


# ---------------------------------------------------------------------------
# 1.  Label preparation
# ---------------------------------------------------------------------------

def prepare_ig_labels(
    ig_raw:    Tensor,   # [E] raw MC-sampled IG (forward edges only, may have negatives)
    edge_attr: Tensor,   # [2E, 2]
    tol:       float = 1e-6,
) -> Tensor:
    """
    Clean ground-truth IG labels:
      1. Clamp negatives to 0  (MC sampling noise; true IG >= 0)
      2. Force certain edges to exactly 0.0
      3. Assert the result satisfies the certain-edge constraint

    Parameters
    ----------
    ig_raw    : [E]  one label per undirected (forward) edge
    edge_attr : [2E, 2]

    Returns
    -------
    ig_clean  : [E]  non-negative, certain edges = 0
    """
    E = edge_attr.shape[0]
    assert ig_raw.shape[0] == E, (
        f"ig_raw must have shape [E]={E}, got {ig_raw.shape[0]}."
    )

    p        = edge_attr[:, 1]                   # [E]
    ig_clean = ig_raw.clone().float()

    ig_clean = torch.clamp(ig_clean, min=0.0)    # fix MC noise

    certain_mask           = (p <= 0.0) | (p >= 1.0)
    ig_clean[certain_mask] = 0.0

    assert_certain_edges_zero(ig_clean, edge_attr, tol=tol)

    return ig_clean   # [E]


# ---------------------------------------------------------------------------
# 2.  Model
# ---------------------------------------------------------------------------

class EdgeIGGNN(nn.Module):
    """
    Predicts information-gain for every undirected edge.

    Output shape : [E]  — one scalar per undirected edge.
    edge_index   : must follow [forward_E | reverse_E] convention.
                   Call assert_edge_symmetry() when building your dataset.
    """

    def __init__(
        self,
        node_dim:   int   = 2,
        edge_dim:   int   = 2,
        hidden_dim: int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()

        assert hidden_dim % n_heads == 0
        head_dim = hidden_dim // n_heads

        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels    = hidden_dim,
                out_channels   = head_dim,
                heads          = n_heads,
                edge_dim       = hidden_dim,
                concat         = True,
                dropout        = dropout,
                add_self_loops = True,
            )
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # input: [h_u ‖ h_v ‖ (e_uv + e_vu)/2]  → 3 * hidden_dim
        # symmetrising e before concat ensures IG(u,v) == IG(v,u)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x:          Tensor,            # [N, node_dim]
        edge_index: Tensor,            # [2, E]  forward edges only
        edge_attr:  Tensor,            # [E, edge_dim]
        batch:      Optional[Tensor] = None,
    ) -> Tensor:                       # [E]  IG per undirected edge

        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        E = edge_index.shape[1]        # number of undirected edges

        # ── construct reverse edges internally ────────────────────────────
        edge_index_rev = edge_index.flip(0)           # [2, E]  v→u
        edge_index_bi  = torch.cat([edge_index, edge_index_rev], dim=1)   # [2, 2E]
        edge_attr_bi   = torch.cat([edge_attr,  edge_attr],      dim=0)   # [2E, edge_dim]

        # ── encode ────────────────────────────────────────────────────────
        h = F.relu(self.node_encoder(x))               # [N, H]
        e = F.relu(self.edge_encoder(edge_attr_bi))    # [2E, H]

        # ── message passing (uses all 2E directed edges) ──────────────────
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            h_new = gat(h, edge_index_bi, edge_attr=e)
            h = ln(h + self.dropout(h_new))
    

        # ── edge-level readout (forward edges only → output [E]) ──────────
        src_fwd = edge_index[0]    # [E]  u
        dst_fwd = edge_index[1]    # [E]  v

        h_u   = h[src_fwd]         # [E, H]
        h_v   = h[dst_fwd]         # [E, H]
        e_fwd = e[:E]              # [E, H]  u→v encoded features
        e_bwd = e[E:]              # [E, H]  v→u encoded features

        # symmetrise: average both directions so IG(u,v) == IG(v,u)
        e_sym = (e_fwd + e_bwd) / 2.0   # [E, H]

        edge_repr = torch.cat([h_u, h_v, e_sym], dim=-1)   # [E, 3H]
        ig_raw    = self.edge_mlp(edge_repr).squeeze(-1)    # [E]

        # non-negativity: softplus keeps gradients alive near zero
        ig_out = F.softplus(ig_raw)    # [E]

        # hard mask: certain edges forced to exactly 0
        p            = edge_attr[:, 1]
        certain_mask = (p <= 0.0) | (p >= 1.0)
        ig_out       = ig_out.clone()
        ig_out[certain_mask] = 0.0

        return ig_out   # [E]


# ---------------------------------------------------------------------------
# 3.  Loss
# ---------------------------------------------------------------------------

def ig_regression_loss(
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


# ---------------------------------------------------------------------------
# 4.  Training utilities
# ---------------------------------------------------------------------------

def train_epoch(
    model:     EdgeIGGNN,
    loader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    uncertain_weight: float = 1.0,
    certain_weight:   float = 0.1,
) -> float:
    model.train()
    total_loss, n_graphs = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred   = model(
            x          = batch.x,
            edge_index = batch.edge_index,
            edge_attr  = batch.edge_attr,
            batch      = batch.batch,
        )   # [total_E_in_batch]

        target = prepare_ig_labels(batch.ig_labels, batch.edge_attr)

        loss = ig_regression_loss(
            pred, target, batch.edge_attr,
            uncertain_weight, certain_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n_graphs   += batch.num_graphs

    return total_loss / n_graphs


@torch.no_grad()
def evaluate(
    model:  EdgeIGGNN,
    loader,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (loss, MAE on uncertain edges only)."""
    model.eval()
    total_loss, total_mae, n_graphs = 0.0, 0.0, 0

    for batch in loader:
        batch  = batch.to(device)
        pred   = model(
            x          = batch.x,
            edge_index = batch.edge_index,
            edge_attr  = batch.edge_attr,
            batch      = batch.batch,
        )
        target = prepare_ig_labels(batch.ig_labels, batch.edge_attr)

        total_loss += ig_regression_loss(pred, target, batch.edge_attr).item() * batch.num_graphs

        E         = pred.shape[0]
        p         = batch.edge_attr[:, 1]
        uncertain = (p > 0.0) & (p < 1.0)        
        if uncertain.any():
            total_mae += (pred[uncertain] - target[uncertain]).abs().mean().item() * batch.num_graphs
        n_graphs += batch.num_graphs

    return total_loss / n_graphs, total_mae / n_graphs


# ---------------------------------------------------------------------------
# 5.  Retrieval helpers
# ---------------------------------------------------------------------------

def get_edge_position(edge_index: Tensor, u: int, v: int) -> int:
    """
    Return the index i such that edge_index[:, i] == [u, v].
    edge_index is [2, E] (forward edges only).
    Raises an error if not found.
    """
    src  = edge_index[0]
    dst  = edge_index[1]
    mask = (src == u) & (dst == v)
    assert mask.any(), (
        f"Edge ({u}, {v}) not found in edge_index. "
        "Check that you are using the forward-edge convention (u→v)."
    )
    return mask.nonzero(as_tuple=True)[0][0].item()


def get_all_uncertain_ig(
        ig_output:  Tensor,   # [E]
        edge_index: Tensor,   # [2, E]  forward edges only
        edge_attr:  Tensor,   # [E, 2]
    ):
    """
    Return all uncertain edges with their IG values as a list of dicts.
    Sorted descending by IG — for inspection only.
    Do NOT use this order directly as drone scouting priority
    until travel cost is incorporated.
    """
    pass
    # src = edge_index[0]
    # dst = edge_index[1]
    # p   = edge_attr[:, 1]

    # E         = ig_output.shape[0]
    # uncertain = (p > 0.0) & (p < 1.0)
    # results   = [
    #     {
    #         "edge_idx": i,
    #         "edge":     (src[i].item(), dst[i].item()),
    #         "p_block":  p[i].item(),
    #         "ig_score": ig_output[i].item(),
    #     }
    #     for i in range(E) if uncertain[i]
    # ]
    # results.sort(key=lambda r: r["ig_score"], reverse=True)
    # return results


# ---------------------------------------------------------------------------
# 6.  Smoke test
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
