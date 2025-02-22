import os
import numpy as np

SBERT_PATH = '/resources/sentence_transformers/'

def get_inter_distances_nodes(nodes, robot_nodes):
    distances = {
        (node1, node2): np.linalg.norm(np.array(node1.location) - np.array(node2.location))
        for node1 in nodes
        for node2 in nodes
    }
    distances.update({
        (robot_node.start, node): np.linalg.norm(np.array(robot_node.start.location) - np.array(node.location))
        for robot_node in robot_nodes
        for node in nodes
    })
    return distances

def get_partial_path_upto_distance(cost_grid, path, distance):
    cost_array = cost_grid[path[0], path[1]]
    cost_array -= distance
    cost_array[cost_array > 0] = 0
    idx = np.argmax(cost_array)
    return path[:, :idx + 1]

def prepare_fcnn_input(graph, containers, objects_to_find):
    graph = graph.copy()

    objs_idx = []
    for obj in objects_to_find:
        idx = graph.add_node({
            'name': obj,
            'type': [0, 0, 0, 1],
        })
        objs_idx.append(idx)

    node_features = compute_node_features(graph.nodes)

    node_features_dict = {}
    for idx, obj in zip(objs_idx, objects_to_find):
        node_feats_input = []
        for container in containers:
            feats = []
            room_idx = graph.get_parent_node_idx(container)
            feats.extend(node_features[room_idx])
            feats.extend(node_features[container])
            feats.extend(node_features[idx])
            node_feats_input.append(feats)
        node_features_dict[obj] = node_feats_input

    return node_features_dict


def compute_node_features(nodes):
    """Get node features for all nodes."""
    features = []
    for node in nodes.values():
        node_feature = np.concatenate((
            get_sentence_embedding(node['name']), node['type']
        ))
        features.append(node_feature)
    return features


def load_sentence_embedding(target_file_name):
    target_dir = os.path.join(SBERT_PATH, 'cache')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # Walk through all directories and files in target_dir
    for root, dirs, files in os.walk(target_dir):
        if target_file_name in files:
            file_path = os.path.join(root, target_file_name)
            if os.path.exists(file_path):
                return np.load(file_path)
    return None

def get_sentence_embedding(sentence):
    loaded_embedding = load_sentence_embedding(sentence + '.npy')
    if loaded_embedding is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SBERT_PATH)
        sentence_embedding = model.encode([sentence])[0]
        file_name = os.path.join(SBERT_PATH, 'cache', sentence + '.npy')
        np.save(file_name, sentence_embedding)
        return sentence_embedding
    else:
        return loaded_embedding
