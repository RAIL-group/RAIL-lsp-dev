import time
import torch
import random
import numpy as np

import procthor
import taskplan
from taskplan.planners.planner import LearnedPlanner


def get_args():
    args = lambda: None
    args.cache_path = '/data/.cache'
    args.current_seed = 7007
    args.resolution = 0.05
    args.save_dir = '/data/test_logs/'
    args.network_file = '/data/taskplan/logs/dbg/gnn.pt'

    args.cost_type = 'learned'
    args.goal_type = '1object'

    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)

    return args


def test_fast_approximation():
    args = get_args()

    thor_data = procthor.ThorInterface(args=args)

    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    whole_graph = thor_data.get_graph()
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=True)
    partial_map.set_room_info(init_robot_pose, thor_data.rooms)

    learned_data = {
        'partial_map': partial_map,
        'learned_net': args.network_file
    }

    t1 = time.time()
    pddl = taskplan.pddl.helper.get_pddl_instance(
        whole_graph=whole_graph,
        map_data=thor_data,
        args=args,
        learned_data=learned_data
    )
    print('PDDL instance generation time:', time.time() - t1)
    raise NotImplementedError


def test_subgoal_selection():
    args = get_args()

    thor_data = procthor.ThorInterface(args=args)

    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    whole_graph = thor_data.get_graph()
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=True)
    partial_map.set_room_info(init_robot_pose, thor_data.rooms)

    destination = partial_map.container_poses[
        partial_map.cnt_node_idx[0]]

    planner = LearnedPlanner(args, partial_map, verbose=True,
                             destination=destination)
    # Initiate planning loop but run for a step
    planning_loop = taskplan.planners.planning_loop.PlanningLoop(
        partial_map=partial_map, robot=init_robot_pose,
        destination=destination, args=args, verbose=True)

    for counter, step_data in enumerate(planning_loop):
        # Update the planner objects
        s_time = time.time()
        planner.update(
            step_data['graph'],
            step_data['subgoals'],
            step_data['robot_pose'])
        print(f"Time taken to update: {time.time() - s_time}")

        # Compute the next subgoal and set to the planning loop
        s_time = time.time()
        chosen_subgoal = planner.compute_selected_subgoal()
        print(f"Time taken to choose subgoal: {time.time() - s_time}")
        planning_loop.set_chosen_subgoal(chosen_subgoal)

    assert hasattr(partial_map, 'room_info')
