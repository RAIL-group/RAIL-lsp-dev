import os
import glob
import torch
import random
import numpy as np
from torch_geometric.data import Data

import learning

SBERT_PATH = '/resources/sentence_transformers/'


def write_datum_to_file(args, datum, counter):
    data_filename = os.path.join('pickles', f'dat_{args.current_seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(
        os.path.join(args.save_dir, data_filename), datum)
    csv_filename = f'{args.data_file_base_name}_{args.current_seed}.csv'
    with open(os.path.join(args.save_dir, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')


def get_data_path_names(args):
    training_data_files = glob.glob(os.path.join(args.data_csv_dir, "*train*.csv"))
    testing_data_files = glob.glob(os.path.join(args.data_csv_dir, "*test*.csv"))
    return training_data_files, testing_data_files


def preprocess_training_data(args=None):
    def make_graph(data):
        data['node_feats'] = torch.tensor(
            np.array(data['node_feats']), dtype=torch.float)
        temp = [[x[0], x[1]] for x in data['edge_index'] if x[0] != x[1]]
        data['edge_index'] = torch.tensor(list(zip(*temp)), dtype=torch.long)
        data['is_subgoal'] = torch.tensor(data['is_subgoal'], dtype=torch.long)
        data['is_target'] = torch.tensor(data['is_target'], dtype=torch.long)
        data['labels'] = torch.tensor(data['labels'], dtype=torch.float)

        tg_GCN_format = Data(x=data['node_feats'],
                             edge_index=data['edge_index'],
                             is_subgoal=data['is_subgoal'],
                             is_target=data['is_target'],
                             # edge_features=data['edge_features'],
                             y=data['labels'])

        result = tg_GCN_format
        return result
    return make_graph


def preprocess_gcn_data(datum):
    data = datum.copy()
    data['edge_index'] = torch.tensor(data['edge_index'], dtype=torch.long)
    data['node_feats'] = torch.tensor(np.array(
        data['node_feats']), dtype=torch.float)
    data['is_subgoal'] = torch.tensor(data['is_subgoal'], dtype=torch.long)
    data['is_target'] = torch.tensor(data['is_target'], dtype=torch.long)
    return data


def prepare_gcn_input(graph, subgoals, target_obj_info):
    # add the target node and connect it to all the subgoal nodes
    # with edges
    graph = graph.copy()
    target_idx = graph.add_node({
        'name': target_obj_info['name'],
        'type': target_obj_info['type'],
    })
    for subgoal in subgoals:
        graph.add_edge(subgoal, target_idx)

    node_features = compute_node_features(graph.nodes)
    gcn_input = {
        'node_feats': node_features,
        'edge_index': np.array(graph.edges).T,
        'is_subgoal': [1 if idx in subgoals else 0 for idx in graph.nodes],
        'is_target': [1 if idx == target_idx else 0 for idx in graph.nodes],
        'target_obj': target_obj_info['name']
    }
    return gcn_input


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


def get_pose_from_coord(coords, nodes):
    coords_list = []
    for k, v in nodes.items():
        coords_list.append((v['position'][0], v['position'][1]))
    if coords in coords_list:
        pos = coords_list.index(coords)
        return pos
    return None


def initialize_environment(cnt_node_idx, seed=0):
    random.seed(seed)
    sample_count = random.randint(1, len(cnt_node_idx))
    undiscovered_cnts = random.sample(cnt_node_idx, sample_count)
    srtd_und_cnts = sorted(undiscovered_cnts)
    return srtd_und_cnts


def get_container_ID(nodes, cnts):
    set_of_ID = set()
    for cnt in cnts:
        set_of_ID.add(nodes[cnt]['id'])
    return set_of_ID


def get_container_pose(cnt_name, partial_map):
    '''This function takes in a container name and the
    partial map as input to return the container pose on the grid'''
    if cnt_name in partial_map.idx_map:
        return partial_map.container_poses[partial_map.idx_map[cnt_name]]
    if cnt_name == 'initial_robot_pose':
        return None
    raise ValueError('The container could not be located on the grid!')


def get_object_to_find_from_plan(plan, partial_map, init_robot_pose):
    '''This function takes in a plan and the partial map as
    input to return the object indices to find, coupled with from
    where and where to locations'''
    find_from_to = {}
    # Robot_poses would be a list of dictionaries in the format
    # (from, to): 'find/move'
    robot_poses = []
    for action in plan:
        if action.name == 'move':
            move_start = action.args[0]
            ms_pose = get_container_pose(move_start, partial_map)
            if ms_pose is None:
                ms_pose = init_robot_pose
            move_end = action.args[1]
            me_pose = get_container_pose(move_end, partial_map)
            if me_pose is None:
                me_pose = init_robot_pose
            robot_poses.append({(ms_pose, me_pose): 'move'})
        elif action.name == 'find':
            obj_name = action.args[0]
            find_start = action.args[1]
            fs_pose = get_container_pose(find_start, partial_map)
            if fs_pose is None:
                fs_pose = init_robot_pose
            find_end = action.args[2]
            fe_pose = get_container_pose(find_end, partial_map)
            if fe_pose is None:
                fe_pose = init_robot_pose
            if obj_name in partial_map.idx_map:
                obj_idx = partial_map.idx_map[obj_name]
                find_from_to[obj_idx] = {
                    'from': fs_pose, 'to': fe_pose}
            robot_poses.append({(fs_pose, fe_pose): 'find'})

    return find_from_to, robot_poses
