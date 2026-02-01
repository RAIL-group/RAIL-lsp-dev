import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
# from torch_geometric.nn import RGCNConv
# from torch_geometric.nn import GATConv
from torch_geometric.nn import NNConv


class BlockScoutNNConv(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, context_dim):
        super().__init__()
        # edge network maps edge_attr → transformation matrix
        edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * node_dim)
        )
        self.conv1 = NNConv(node_dim, hidden_dim, edge_net, aggr='mean')
        self.conv2 = NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean')

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, block_mask, ugv_pose):
        h = torch.relu(self.conv1(x, edge_index, edge_attr))
        h = torch.relu(self.conv2(h, edge_index, edge_attr))
        context = ugv_pose  # assuming ugv_pose is already the correct context
        context = context.repeat(h.size(0), 1)
        combined = torch.cat([h, context], dim=-1)
        values = self.value_head(combined)
        return values[block_mask]

# def convert_to_pyg_data(graph, action, robot_pos, y):
#     vertices = graph.vertices
#     pois = graph.pois

#     num_vertices = len(vertices)
#     num_pois = len(pois)
#     num_nodes = num_vertices + num_pois

#     # ----- Node features -----
#     node_features = []
#     for v in vertices:
#         node_features.append([v.coord[0], v.coord[1], v.block_prob, 0])
#     for p in pois:
#         node_features.append([p.coord[0], p.coord[1], p.block_prob, 1])
#     node_features = torch.tensor(node_features, dtype=torch.float)
    
#     action_feature = [[poi.coord[0], poi.coord[1], p.block_prob, 1] for poi in pois if p.id == action]
#     action_feature = torch.tensor(action_feature, dtype=torch.float)

#     # ----- Edge list and edge attributes -----
#     edge_list, edge_attr = [], []

#     for p in pois:
#         v1, v2 = p.neighbors  # ensure this exists
#         i, j = v1.id, v2.id
#         # Distances between POI and intersections
#         dist1 = np.linalg.norm(np.array(v1.coord) - np.array(p.coord))
#         dist2 = np.linalg.norm(np.array(v2.coord) - np.array(p.coord))

#         edge_list += [[i, p.id], [p.id, i]]
#         edge_attr += [[dist1], [dist1]]

#         edge_list += [[j, p.id], [p.id, j]]
#         edge_attr += [[dist2], [dist2]]

#     edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
#     edge_attr = torch.tensor(edge_attr, dtype=torch.float)

#     # ----- Masks -----
#     block_mask = torch.zeros(num_nodes, dtype=torch.bool)
#     block_mask[num_vertices:] = True

#     # ----- Context -----
#     if robot_pos.get('on_edge', False):
#         v1, v2 = robot_pos['edge']
#         s = robot_pos['s']
#         coord = (1 - s) * np.array(v1.coord) + s * np.array(v2.coord)
#     else:
#         coord = np.array(robot_pos['coord'])
#     robot_state = torch.tensor([coord[0], coord[1]], dtype=torch.float).unsqueeze(0)

#     # ----- Labels -----
#     y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

#     return Data(
#         x=node_features,
#         edge_index=edge_index,
#         edge_attr=edge_attr,
#         action=action_feature,
#         blockpoint_mask=block_mask,
#         robot_state=robot_state,
#         y=y
#     )