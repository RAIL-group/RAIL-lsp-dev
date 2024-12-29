import os
import glob
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data

import learning
import procthor


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
    data['edge_data'] = torch.tensor(data['edge_index'], dtype=torch.long)
    data['latent_features'] = torch.tensor(np.array(
        data['node_feats']), dtype=torch.float)
    data['is_subgoal'] = torch.tensor(data['is_subgoal'], dtype=torch.long)
    data['is_target'] = torch.tensor(data['is_target'], dtype=torch.long)
    return data


def get_pose_from_coord(coords, whole_graph):
    coords_list = []
    for node in whole_graph['node_coords']:
        coords_list.append(tuple(
            [whole_graph['node_coords'][node][0],
             whole_graph['node_coords'][node][1]]))
    if coords in coords_list:
        pos = coords_list.index(coords)
        return pos
    return None


def initialize_environment(cnt_node_idx, seed=0):
    random.seed(seed)
    # at least 80% of the containers should be undiscovered
    cnt_count = len(cnt_node_idx)
    sample_count = random.randint(math.ceil(0.8 * cnt_count), cnt_count)
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


def get_coffee_objects():
    '''This function returns a dictionary of objects and their
    possible spawn locations in the environment.'''
    obj_loc_dict = {
        'waterbottle': ['fridge', 'diningtable'],
        'coffeegrinds': ['diningtable', 'countertop', 'shelvingunit']
    }
    return obj_loc_dict


def get_robots_room_coords(occupancy_grid, robot_pose, rooms, return_idx=False):
    '''This function takes in the room data, robot pose and the
    occupancy grid as input to return the room coordinates closest
    to the robot pose'''
    t_cost = 9999999
    room_coords = None
    for idx, room in enumerate(rooms):
        room_coords = room['position']
        cost = procthor.utils.get_cost(occupancy_grid, robot_pose, room_coords)
        if cost < t_cost:
            t_cost = cost
            room_coords = room['position']
    if return_idx:
        return idx + 1
    return room_coords


def check_skip_protocol(args):
    # load the ignore list from args.fail_log
    if hasattr(args, 'fail_log') and args.fail_log:
        IGNORE_LIST = load_fail_log(args.fail_log)
        if args.current_seed in IGNORE_LIST:
            plt.title("Skipping due to prior failure!")
            plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=100)
            exit()


def load_fail_log(fail_log):
    failure_seeds = []
    if os.path.exists(fail_log):
        with open(fail_log, 'r') as lines:
            for line in lines:
                parts = line.split("]")
                failure_seeds.append(int(parts[0][1:]))

    return failure_seeds


def save_fail_log(fail_log, seed, error_msg=''):
    # open the file in append mode
    with open(fail_log, 'a') as f:
        f.write(f"[{seed}] {error_msg}\n")


def get_action_costs():
    scale = 1
    action_costs = {
        'pour-water': 100 * scale,
        'pour-coffee': 100 * scale,
        'make-coffee': 100 * scale,
        'boil': 100 * scale,
        'peel': 100 * scale,
        'toast': 100 * scale,
        'pick': 100 * scale,
        'place': 100 * scale,
        'find': 0 * scale
    }
    return action_costs


def check_pddl_validity(pddl, args):
    if not pddl['goal']:
        error_msg = "No valid goal settings found!"
        save_fail_log(args.fail_log, args.current_seed, error_msg)
        plt.title(error_msg)
        plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=100)
        exit()


def check_plan_validity(plan, args, cost_str=None):
    if not plan:
        if plan == []:
            error_msg = "Goal already satisfied with initial settings!"
        elif cost_str:
            error_msg = f"==== Replanning Failed [{cost_str}] ===="
        elif plan is None:
            error_msg = "No valid plan found with initial settings!"
        save_fail_log(args.fail_log, args.current_seed, error_msg)
        plt.title(error_msg)
        plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=100)
        exit()


def get_cost_string(args):
    if args.logfile_name == 'task_learned_logfile.txt':
        cost_str = 'learned'
    elif args.logfile_name == 'task_optimistic_greedy_logfile.txt':
        cost_str = 'optimistic_greedy'
    elif args.logfile_name == 'task_pessimistic_greedy_logfile.txt':
        cost_str = 'pessimistic_greedy'
    elif args.logfile_name == 'task_optimistic_lsp_logfile.txt':
        cost_str = 'optimistic_lsp'
    elif args.logfile_name == 'task_pessimistic_lsp_logfile.txt':
        cost_str = 'pessimistic_lsp'
    elif args.logfile_name == 'task_optimistic_oracle_logfile.txt':
        cost_str = 'optimistic_oracle'
    elif args.logfile_name == 'task_pessimistic_oracle_logfile.txt':
        cost_str = 'pessimistic_oracle'
    elif args.logfile_name == 'task_oracle_logfile.txt':
        cost_str = 'oracle'
    return cost_str
