import time
import torch
import random
import numpy as np

from procthor import procthor
import taskplan
from taskplan.planners.planner import LearnedPlanner


def get_args():
    args = lambda: None
    args.cache_path = '/data/.cache'
    args.current_seed = 0
    args.resolution = 0.05
    args.save_dir = '/data/test_logs/'
    args.network_file = '/data/taskplan/logs/dbg/fcnn.pt'

    args.cost_type = 'learned'

    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)

    return args


def test_bowl():
    args = get_args()
    args.current_seed = 6200

    thor_data = procthor.ThorInterface(args=args)

    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    whole_graph = thor_data.get_graph()
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=False)

    init_subgoals_idx = taskplan.utilities.utils.initialize_environment(
            whole_graph['cnt_node_idx'], args.current_seed)

    # partial_map.target = 'do not know yet find the seend and set the target'
    target_obj = partial_map.idx_map['bowl|surface|6|39']
    partial_map.target_obj = target_obj
    partial_map.initialize_graph_and_subgoals()
    graph, subgoals = partial_map.update_graph_and_subgoals(
        subgoals=init_subgoals_idx)

    goals = [partial_map.org_node_names[goal]
             for goal in partial_map.target_container
             if goal in subgoals]
    print(f'[{args.current_seed}] Target object: {partial_map.org_node_names[partial_map.target_obj]}')
    print(f'Target is hidden in: {goals}')

    planner = LearnedPlanner(args, partial_map, verbose=False)
    planner.update(graph, subgoals, init_robot_pose)
    print_predictions(planner, partial_map)

    args.current_seed = 6100

    thor_data = procthor.ThorInterface(args=args)

    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    whole_graph = thor_data.get_graph()
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=False)

    init_subgoals_idx = taskplan.utilities.utils.initialize_environment(
            whole_graph['cnt_node_idx'], args.current_seed)

    # partial_map.target = 'do not know yet find the seend and set the target'
    target_obj = partial_map.idx_map['bowl|surface|7|35']
    partial_map.target_obj = target_obj
    partial_map.initialize_graph_and_subgoals()
    graph, subgoals = partial_map.update_graph_and_subgoals(
        subgoals=init_subgoals_idx)

    goals = [partial_map.org_node_names[goal]
             for goal in partial_map.target_container
             if goal in subgoals]
    print(f'[{args.current_seed} Target object: {partial_map.org_node_names[partial_map.target_obj]}')
    print(f'Target is hidden in: {goals}')

    planner = LearnedPlanner(args, partial_map, verbose=False)
    planner.update(graph, subgoals, init_robot_pose)
    print_predictions(planner, partial_map)
    raise NotImplementedError


def test_pillow():
    args = get_args()
    args.current_seed = 6200

    thor_data = procthor.ThorInterface(args=args)

    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    whole_graph = thor_data.get_graph()
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=False)

    init_subgoals_idx = taskplan.utilities.utils.initialize_environment(
            whole_graph['cnt_node_idx'], args.current_seed)

    # partial_map.target = 'do not know yet find the seend and set the target'
    target_obj = partial_map.idx_map['pillow|7|0|2']
    partial_map.target_obj = target_obj
    partial_map.initialize_graph_and_subgoals()
    graph, subgoals = partial_map.update_graph_and_subgoals(
        subgoals=init_subgoals_idx)

    goals = [partial_map.org_node_names[goal]
             for goal in partial_map.target_container
             if goal in subgoals]
    print(f'[seed: {args.current_seed}] Target object: {partial_map.org_node_names[partial_map.target_obj]}')
    print(f'Target is hidden in: {goals}')

    planner = LearnedPlanner(args, partial_map, verbose=False)
    planner.update(graph, subgoals, init_robot_pose)
    print_predictions(planner, partial_map)

    args.current_seed = 6100

    thor_data = procthor.ThorInterface(args=args)

    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    whole_graph = thor_data.get_graph()
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=False)

    init_subgoals_idx = taskplan.utilities.utils.initialize_environment(
            whole_graph['cnt_node_idx'], args.current_seed)

    # partial_map.target = 'do not know yet find the seend and set the target'
    target_obj = partial_map.idx_map['pillow|4|2|2']
    partial_map.target_obj = target_obj
    partial_map.initialize_graph_and_subgoals()
    graph, subgoals = partial_map.update_graph_and_subgoals(
        subgoals=init_subgoals_idx)

    goals = [partial_map.org_node_names[goal]
             for goal in partial_map.target_container
             if goal in subgoals]
    print(f'[seed: {args.current_seed}] Target object: {partial_map.org_node_names[partial_map.target_obj]}')
    print(f'Target is hidden in: {goals}')

    planner = LearnedPlanner(args, partial_map, verbose=False)
    planner.update(graph, subgoals, init_robot_pose)
    print_predictions(planner, partial_map)
    raise NotImplementedError


def print_predictions(planner, partial_map):
    for subgoal in planner.subgoals:
        if subgoal.value in partial_map.target_container:
            print(
                f'Ps={subgoal.prob_feasible:6.4f} | '
                f'at={partial_map.org_node_names[subgoal.value]}'
            )
    print("-----------------------------")
    for subgoal in planner.subgoals:
        if subgoal.value not in partial_map.target_container:
            print(
                f'Ps={subgoal.prob_feasible:6.4f} | '
                f'at={partial_map.org_node_names[subgoal.value]}'
            )
