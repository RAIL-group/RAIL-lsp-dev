import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

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


def prepare_lspllm_input(graph, subgoals, target_obj_info):
    datum = {
        'graph': graph,
        'subgoals': subgoals,
        'target_obj_info': target_obj_info,
    }
    return datum


def compute_node_features(nodes):
    """Get node features for all nodes."""
    features = []
    for node in nodes.values():
        node_feature = np.concatenate((
            get_sbert_embedding(node['name']), node['type']
        ))
        features.append(node_feature)
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
