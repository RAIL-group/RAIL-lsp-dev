import numpy as np
import argparse
import procthor
from procthor.simulators import SceneGraphSimulator
import mr_task
from mr_task.core import Node
import matplotlib.pyplot as plt
from common import Pose


def _setup(args):
    thor_interface = procthor.ThorInterface(args)
    known_graph, known_grid, _, target_objs_info = thor_interface.gen_map_and_poses(num_objects=3)
    specification = mr_task.specification.get_random_specification(objects=[obj['name'] for obj in target_objs_info],
                                                                   seed=args.current_seed)
    mrtask_planner = mr_task.planner.LearnedMRTaskPlanner(args, specification)
    simulator = SceneGraphSimulator(args=args, known_graph=known_graph,
                    target_objs_info=target_objs_info, known_grid=known_grid, thor_interface=thor_interface)

    graph, containers_idx = simulator.initialize_graph_and_containers()

    unexplored_container_nodes = [Node(is_subgoal=True,
                                       name=idx,
                                       location=simulator.known_graph.get_node_position_by_idx(idx))
                                  for idx in containers_idx]
    mrtask_planner.update(
        observations= {'observed_graph': graph},
        robot_poses=[],
        explored_container_nodes=[],
        unexplored_container_nodes=unexplored_container_nodes,
        objects_found=(),
    )
    # print these in a file
    print("Seed: ", args.current_seed)
    with open(f'{args.save_dir}/network_output.txt', 'a') as f:
        f.write('----------------------------------\n')
        f.write(f"Seed: {args.current_seed}\n")
        f.write(f"Specification: {specification}\n")
        for key, val in mrtask_planner.node_prop_dict.items():
            f.write(f"{graph.get_node_name_by_idx(key[0].name)} {key[1]} PS: {val[0]:.2f}\n")
        f.write('----------------------------------\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/mr_task')
    parser.add_argument('--network_file', type=str, default='/data/mr_task')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--resolution', type=float, default=0.05)
    args = parser.parse_args()
    for i in range(10):
        args.current_seed = args.seed + i
        _setup(args)
