import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import lsp
from common import Pose
import object_search

SBERT_PATH = '/resources/sentence_transformers/'


def prepare_fcnn_input(graph, subgoals, target_obj_info):
    graph = graph.copy()

    target_obj_idx = graph.add_node({
        'name': target_obj_info['name'],
        'type': [0, 0, 0, 1],
    })
    node_features = compute_node_features(graph.nodes)

    node_feats_input = []
    for subgoal in subgoals:
        feats = []
        room_idx = graph.get_parent_node_idx(subgoal.id)
        feats.extend(node_features[room_idx])
        feats.extend(node_features[subgoal.id])
        feats.extend(node_features[target_obj_idx])
        node_feats_input.append(feats)
    datum = {'node_feats': np.array(node_feats_input)}
    return datum


def get_frontier_location_room_name(graph, grid, frontier, room_indices):
    """Get the room name where the frontier is located based on the nearest room."""
    frontier_pose = Pose(*frontier.get_frontier_point())
    rooms = [
        object_search.core.Subgoal(idx, graph.get_node_position_by_idx(idx)[:2])
        for idx in room_indices
    ]
    room_distances = lsp.core.get_robot_distances(grid, frontier_pose, rooms)
    nearest_room = min(room_distances, key=room_distances.get)
    room_name = graph.get_node_name_by_idx(nearest_room.id)
    frontier.room_name = room_name
    return room_name


def prepare_fcnn_input_frontiers(graph, grid, subgoals, target_obj_info):
    graph = graph.copy()

    room_indices = graph.room_indices
    for subgoal in subgoals:
        if isinstance(subgoal, lsp.core.Frontier):
            subgoal.id = graph.add_node({
                'name': get_frontier_location_room_name(graph, grid, subgoal, room_indices),
                'type': [0, 1, 0, 0],
            })

    target_obj_idx = graph.add_node({
        'name': target_obj_info['name'],
        'type': [0, 0, 0, 1],
    })
    node_features = compute_node_features(graph.nodes)

    node_feats_input = []
    for subgoal in subgoals:
        feats = []
        if isinstance(subgoal, lsp.core.Frontier):
            feats.extend(node_features[subgoal.id])
        else:
            room_idx = graph.get_parent_node_idx(subgoal.id)
            feats.extend(node_features[room_idx])
        feats.extend(node_features[subgoal.id])
        feats.extend(node_features[target_obj_idx])
        node_feats_input.append(feats)
    datum = {'node_feats': np.array(node_feats_input)}
    return datum


def prepare_lspllm_input(graph, subgoals, target_obj_info):
    datum = {
        'graph': graph,
        'subgoals': subgoals,
        'target_obj_info': target_obj_info,
    }
    return datum


def compute_node_features(nodes):
    """Get node features for all nodes."""
    features = {}
    for idx, node in nodes.items():
        node_feature = np.concatenate((
            get_sbert_embedding(node['name']), node['type']
        ))
        features[idx] = node_feature
    return features


def get_sbert_embedding(text):
    embedding_cache_file = Path(SBERT_PATH) / f'cache/{text}.npy'
    if embedding_cache_file.exists():
        return np.load(embedding_cache_file)

    model = SentenceTransformer(SBERT_PATH)
    embedding = model.encode([text])[0]
    embedding_cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_cache_file, embedding)
    return embedding
