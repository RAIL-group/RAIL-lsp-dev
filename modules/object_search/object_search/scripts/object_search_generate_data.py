import argparse
import matplotlib.pyplot as plt
import procthor
from procthor.simulators import SceneGraphSimulator
from object_search.learning.utils import prepare_fcnn_input
from object_search.core import Subgoal
from lsp.utils.data import write_training_data_to_pickle
from pathlib import Path


def get_all_objects_info(known_graph):
    object_name_to_idxs = {}
    for idx in known_graph.object_indices:
        name = known_graph.get_node_name_by_idx(idx)
        if name not in object_name_to_idxs.keys():
            object_name_to_idxs[name] = [idx]
        else:
            object_name_to_idxs[name].append(idx)

    target_obj_names = object_name_to_idxs.keys()
    target_objs_info = []
    for name in target_obj_names:
        idxs = object_name_to_idxs[name]
        container_idxs = [known_graph.get_parent_node_idx(idx) for idx in idxs]
        node_type = known_graph.nodes[idxs[0]]['type']
        target_objs_info.append({
            'name': name,
            'idxs': object_name_to_idxs[name],
            'type': node_type,
            'container_idxs': container_idxs
        })

    return target_objs_info


def get_training_data(known_graph, containers, target_obj_info):
    training_data = []
    subgoals = [Subgoal(idx, known_graph.get_node_position_by_idx(idx)) for idx in containers]
    input_data = prepare_fcnn_input(known_graph, subgoals, target_obj_info)
    for i, subgoal in enumerate(subgoals):
        datum = {
            'node_feats': input_data['node_feats'][i],
        }
        objects_idx = known_graph.get_adjacent_nodes_idx(subgoal.id, filter_by_type=3)
        objects_name = [known_graph.get_node_name_by_idx(idx) for idx in objects_idx]
        if target_obj_info['name'] in objects_name:
            datum['labels'] = 1
        else:
            datum['labels'] = 0
        training_data.append(datum)
    return training_data


def generate_data(args):
    thor_interface = procthor.ThorInterface(args)
    known_graph, known_grid, _, _ = thor_interface.gen_map_and_poses()
    target_objects = get_all_objects_info(known_graph)

    for counter, target_obj_info in enumerate(target_objects):
        simulator = SceneGraphSimulator(known_graph,
                                        args,
                                        target_obj_info,
                                        known_grid,
                                        thor_interface)
        _, _, containers = simulator.initialize_graph_grid_and_containers()

        training_data = get_training_data(known_graph, containers, target_obj_info)
        write_training_data_to_pickle(training_data, counter, args)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    procthor.plotting.plot_graph_on_grid(ax, known_grid, known_graph)
    plt.title(f'Seed: {args.current_seed}')
    plt.savefig(Path(args.save_dir) / f'data_collect_plots/{args.data_file_base_name}_{args.current_seed}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--current_seed', type=int, default=0)
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--data_file_base_name', type=str, default='data_training')

    args = parser.parse_args()
    generate_data(args)
