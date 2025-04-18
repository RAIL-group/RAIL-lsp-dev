import os
import time
import torch
import pytest
import random
import numpy as np
import matplotlib.pyplot as plt
from pddlstream.algorithms.search import solve_from_pddl

import procthor
import taskplan
from taskplan.planners.planner import KnownPlanner
from taskplan.utilities.utils import get_container_pose


def get_args():
    args = lambda: None
    args.current_seed = 2
    args.resolution = 0.05
    args.save_dir = '/data/test_logs/'
    args.image_filename = 'tester.png'
    args.network_file = '/data/taskplan/logs/dbg/fcnn.pt'
    args.cache_path = '/data/.cache'
    args.cost_type = None
    args.goal_type = '1object'

    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)

    return args

@pytest.mark.timeout(30)
def test_learned_plan():
    args = get_args()
    if not os.path.exists(args.network_file):
        pytest.xfail(f"Network file not found: {args.network_file}")

    # Load data for a given seed
    thor_data = procthor.procthor.ThorInterface(args=args)

    # Get the occupancy grid from data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph, grid)
    partial_map.set_room_info(init_robot_pose, thor_data.rooms)

    args.cost_type = 'learned'
    if args.cost_type == 'learned':
        learned_data = {
            'partial_map': partial_map,
            'learned_net': args.network_file
        }
    else:
        learned_data = None
    # Instantiate PDDL for this map
    pddl = taskplan.pddl.helper.get_pddl_instance(
        whole_graph=whole_graph,
        map_data=thor_data,
        args=args,
        learned_data=learned_data
    )
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])

    if plan:
        for idx, p in enumerate(plan):
            print(idx, p)


@pytest.mark.timeout(15)
def test_place_task():
    args = get_args()

    # Load data for a given seed
    thor_data = procthor.procthor.ThorInterface(args=args)

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Instantiate PDDL for this map
    pddl = taskplan.pddl.helper.get_pddl_instance(
        whole_graph=whole_graph,
        map_data=thor_data,
        args=args
    )
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
    if plan:
        for p in plan:
            print(p)
        print(cost)


# @pytest.mark.timeout(50)
def test_replan():
    args = get_args()
    args.current_seed = 7004
    args.goal_type = 'coffee'
    args.cost_type = 'known'

    obj_loc_dict = {
        'waterbottle': ['fridge', 'diningtable'],
        'coffeegrinds': ['diningtable', 'countertop', 'shelvingunit']
    }
    # Load data for a given seed
    thor_data = procthor.procthor.ThorInterface(args=args, preprocess=obj_loc_dict)

    # Get the occupancy grid from data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=True)

    # Instantiate PDDL for this map
    pddl = taskplan.pddl.helper.get_pddl_instance(
        whole_graph=whole_graph,
        map_data=thor_data,
        args=args
    )

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
    executed_actions = []
    robot_poses = [init_robot_pose]
    cost_str = 'Known'
    while plan:
        for action in plan:
            print(action)
            executed_actions.append(action)
            if action.name == 'move':
                move_start = action.args[0]
                ms_pose = get_container_pose(move_start, partial_map)
                if ms_pose is None:
                    ms_pose = init_robot_pose
                move_end = action.args[1]
                me_pose = get_container_pose(move_end, partial_map)
                if me_pose is None:
                    me_pose = init_robot_pose

                # Update problem for move action.
                # (rob-at move_end)
                robot_poses.append(me_pose)
                pddl['problem'] = taskplan.pddl.helper.update_problem_move(
                    pddl['problem'], move_end)
                # Finally replan
                plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
                break
            elif action.name == 'pick':
                object_name = action.args[0]
                pick_at = action.args[1]
                pick_pose = get_container_pose(pick_at, partial_map)
                if pick_pose is None:
                    pick_pose = init_robot_pose
                # Update problem for pick action.
                # (not (hand-is-free))
                # (not (is-at object location))
                # (is holding object)
                pddl['problem'] = taskplan.pddl.helper.update_problem_pick(
                    pddl['problem'], object_name, pick_at)
                # Finally replan
                plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
                break
            elif action.name == 'place':
                object_name = action.args[0]
                place_at = action.args[1]
                place_pose = get_container_pose(place_at, partial_map)
                if place_pose is None:
                    place_pose = init_robot_pose
                # Update problem for place action.
                # (hand-is-free)
                # (is-at object location)
                # (not (is holding object))
                pddl['problem'] = taskplan.pddl.helper.update_problem_place(
                    pddl['problem'], object_name, place_at)
                # Finally replan
                plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
                break
            elif action.name == 'pour-water':
                pour_from = action.args[0]
                pour_to = action.args[1]
                # Update problem for pour-water action.
                # (filled-with-water ?pour_to)
                # (not (filled-with-water ?pour_from))
                # (not (ban-move))
                pddl['problem'] = taskplan.pddl.helper.update_problem_pourwater(
                    pddl['problem'], pour_from, pour_to)
                # Finally replan
                plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
                break
            elif action.name == 'pour-coffee':
                pour_from = action.args[0]
                pour_to = action.args[1]
                # Update problem for pour-coffee action.
                # (filled-with-coffee ?pour_to)
                # (not (filled-with-coffee ?pour_from))
                # (not (ban-move))
                pddl['problem'] = taskplan.pddl.helper.update_problem_pourcoffee(
                    pddl['problem'], pour_from, pour_to)
                # Finally replan
                plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
                break
            elif action.name == 'make-coffee':
                receptacle = action.args[1]
                # Update problem for make-coffee action.
                # (filled-with-coffee ?receptacle)
                # (not (filled-with-water ?receptacle))
                # (not (ban-move))
                pddl['problem'] = taskplan.pddl.helper.update_problem_makecoffee(
                    pddl['problem'], receptacle)
                # Finally replan
                plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
            elif action.name == 'find':
                obj_name = action.args[0]
                obj_idx = partial_map.idx_map[obj_name]
                find_start = action.args[1]
                fs_pose = get_container_pose(find_start, partial_map)
                if fs_pose is None:
                    fs_pose = init_robot_pose
                find_end = action.args[2]
                fe_pose = get_container_pose(find_end, partial_map)
                if fe_pose is None:
                    fe_pose = init_robot_pose

                # Initialize the partial map
                partial_map.target_obj = obj_idx
                # Over here initiate the planner
                planner = KnownPlanner(args, partial_map,
                                       destination=fe_pose)
                # Initiate planning loop but run for a step
                planning_loop = taskplan.planners.planning_loop.PlanningLoop(
                    partial_map=partial_map, robot=fs_pose,
                    destination=fe_pose, args=args, verbose=True)

                planning_loop.subgoals = pddl['subgoals'].copy()
                explored_loc = None

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

                    explored_loc = chosen_subgoal.value
                    pddl['subgoals'].remove(chosen_subgoal.value)
                    break  # Run the loop only exploring one containers

                # Get which container was chosen to explore
                # Get the objects that are connected to that container
                idx2assetID = {partial_map.idx_map[assetID]: assetID for assetID in partial_map.idx_map}

                connection_idx = [
                    partial_map.org_edge_index[1][idx]
                    for idx, value in enumerate(partial_map.org_edge_index[0])
                    if value == explored_loc
                ]
                found_objects = [
                    idx2assetID[con_idx]
                    for con_idx in connection_idx
                ]
                found_at = idx2assetID[explored_loc]

                # Update problem for find action.
                # (rob-at {found_at})
                # For all found_objs (is-located obj)
                #                   (is-at obj found_at)
                # add all the contents of that container in the known space [set as located and where]
                robot_poses.append(partial_map.container_poses[explored_loc])
                pddl['problem'] = taskplan.pddl.helper.update_problem_move(
                    pddl['problem'], found_at)
                for obj in found_objects:
                    pddl['problem'] = taskplan.pddl.helper.update_problem_find(
                        pddl['problem'], obj, found_at)

                # Finally replan
                print('Replanning .. .. ..')
                plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
                break

    print('\nThe final plan as executed:')
    for action in executed_actions:
        print(action)

    distance, trajectory = taskplan.core.compute_path_cost(partial_map.grid, robot_poses)
    print(f"Planning cost: {distance}")

    plt.clf()
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"{pddl['goal']} - seed: [{args.current_seed}]", fontsize=6)

    # 1 plot the plan
    plt.subplot(221)
    taskplan.plotting.plot_plan(plan=executed_actions)

    # 2 plot the whole graph
    plt.subplot(222)
    plt.title('Whole scene graph', fontsize=6)
    plt.imshow(whole_graph['graph_image'])
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])

    plt.subplot(223)
    # 3 plot the top-dwon-view
    top_down_frame = thor_data.get_top_down_frame()
    plt.imshow(top_down_frame)
    plt.title('Top-down view of the map', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.subplot(224)
    # 4 plot the grid with trajectory viridis color
    plotting_grid = procthor.plotting.make_blank_grid(
        np.transpose(grid)
    )
    plt.imshow(plotting_grid)
    plt.title(f"{cost_str} Cost: {distance:0.3f}", fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    viridis_cmap = plt.get_cmap('viridis')

    colors = np.linspace(0, 1, len(trajectory[0]))
    line_colors = viridis_cmap(colors)

    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)

    # Hide box and ticks
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    assert len(executed_actions) > 0
    plt.savefig(f'/data/test_logs/replan_{args.current_seed}.png', dpi=400)


# @pytest.mark.timeout(15)
def test_custom_goal():
    args = get_args()
    args.current_seed = 7004
    args.goal_type = 'coffee'
    args.cost_type = 'known'

    obj_loc_dict = {
        'waterbottle': ['fridge', 'diningtable'],
        'coffeegrinds': ['diningtable', 'countertop', 'shelvingunit']
    }
    # Load data for a given seed
    thor_data = procthor.procthor.ThorInterface(args=args, preprocess=obj_loc_dict)

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Instantiate PDDL for this map
    pddl = taskplan.pddl.helper.get_pddl_instance(
        whole_graph=whole_graph,
        map_data=thor_data,
        args=args
    )

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
    assert len(plan) > 0
