import os
import time
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pddlstream.algorithms.search import solve_from_pddl

import procthor
import taskplan
from taskplan.planners.planner import ClosestActionPlanner, LearnedPlanner, KnownPlanner
from taskplan.utilities.utils import get_container_pose


def evaluate_main(args):
    args = get_args()

    # get the custom coffee related objects
    if 'coffee' in args.goal_type:
        coffee_objects = taskplan.utilities.utils.get_coffee_objects()
        preprocess = coffee_objects
    else:
        preprocess = True
    # Load data for a given seed
    thor_data = procthor.ThorInterface(args=args, preprocess=preprocess)

    # Get the occupancy grid from data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=True)
    partial_map.set_room_info(init_robot_pose, thor_data.rooms)

    if args.cost_type == 'learned':
        learned_data = {
            'partial_map': partial_map,
            'initial_robot_pose': init_robot_pose,
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

    if not pddl['goal']:
        plt.title("No valid goal found!")
        plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=100)
        exit()

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

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'],
                                 planner=pddl['planner'], max_planner_time=300)
    executed_actions = []
    robot_poses = [init_robot_pose]

    if not plan:
        if plan == []:
            plt.title("Goal already satisfied with initial settings!")
        elif plan is None:
            plt.title("No valid plan found with initial settings!")
        plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=100)
        exit()

    # Intialize logfile
    logfile = os.path.join(args.save_dir, args.logfile_name)
    with open(logfile, "a+") as f:
        f.write(f"LOG: {args.current_seed}\n")

    while plan:
        for action_idx, action in enumerate(plan):
            # Break loop at the end of plan
            if action_idx == len(plan) - 1:
                plan = []
            executed_actions.append(action)
            if action.name == 'pour-water':
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
            elif action.name == 'move':
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
            elif action.name == 'boil':
                obj1_name = action.args[0]
                # Update problem for boil action.
                # (is-boiled obj1_name)
                pddl['problem'] = taskplan.pddl.helper.update_problem_boil(
                    pddl['problem'], obj1_name)
            elif action.name == 'peel':
                obj1_name = action.args[0]
                # Update problem for peel action.
                # (is-peeled obj1_name)
                pddl['problem'] = taskplan.pddl.helper.update_problem_peel(
                    pddl['problem'], obj1_name)
            elif action.name == 'toast':
                obj1_name = action.args[0]
                # Update problem for toast action.
                # (is-toasted obj1_name)
                pddl['problem'] = taskplan.pddl.helper.update_problem_toast(
                    pddl['problem'], obj1_name)
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
                if 'greedy' in args.logfile_name:
                    planner = ClosestActionPlanner(args, partial_map,
                                                   destination=fe_pose)
                elif 'oracle' in args.logfile_name:
                    planner = KnownPlanner(args, partial_map,
                                           destination=fe_pose)
                else:
                    planner = LearnedPlanner(args, partial_map, verbose=True,
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
                plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'],
                                             planner=pddl['planner'], max_planner_time=300)
                if plan is None:
                    plt.title("==== Replanning Failed ====")
                    plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=100)
                    exit()
                break

    distance, trajectory = taskplan.core.compute_path_cost(partial_map.grid, robot_poses)
    print(f"Planning cost: {distance}")

    with open(logfile, "a+") as f:
        # err_str = '' if did_succeed else '[ERR]'
        f.write(f"[Learn] s: {args.current_seed:4d}"
                f" | {cost_str}: {distance:0.3f}\n")

    plt.clf()
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"{pddl['goal']} - seed: [{args.current_seed}]", fontsize=6)

    plt.subplot(231)
    # 0 plot the plan
    taskplan.plotting.plot_plan(plan=executed_actions)

    plt.subplot(232)
    # 1 plot the whole graph
    plt.title('Whole scene graph', fontsize=6)
    graph_image = whole_graph['graph_image']
    plt.imshow(graph_image)
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])

    viridis_cmap = plt.get_cmap('viridis')

    colors = np.linspace(0, 1, len(trajectory[0]))
    line_colors = viridis_cmap(colors)

    plt.subplot(233)
    # 2 plot the top-dwon-view
    top_down_frame = thor_data.get_top_down_frame()
    offset = thor_data.plot_offset
    extent = thor_data.plot_extent
    plt.imshow(top_down_frame, extent=extent)
    plt.title('Top-down view of the map', fontsize=6)
    for idx, x in enumerate(trajectory[0]):
        x = x + offset[0]
        y = trajectory[1][idx] + offset[1]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.subplot(234)
    # 3 plot the graph overlaied image
    procthor.plotting.plot_graph_on_grid(grid, whole_graph)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title('Graph overlaied occupancy grid', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.subplot(235)
    # 4 plot the graph overlaied image
    plotting_grid = procthor.plotting.make_plotting_grid(np.transpose(grid))
    plt.imshow(plotting_grid)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title('Path overlaied occupancy grid', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)

    plt.subplot(236)
    # 5 plot the grid with trajectory viridis color
    plotting_grid = procthor.plotting.make_blank_grid(
        np.transpose(grid)
    )
    plt.imshow(plotting_grid)
    plt.title(f"{cost_str} Cost: {distance:0.3f}", fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)

    # Hide box and ticks
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=1200)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for Task Planning under uncertainty"
    )
    parser.add_argument('--current_seed', type=int, required=True)
    parser.add_argument('--image_filename', type=str, required=True)
    parser.add_argument('--logfile_name', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--resolution', type=float, required=True)
    parser.add_argument('--network_file', type=str, required=False)
    parser.add_argument('--goal_type', type=str, required=False)
    parser.add_argument('--cost_type', type=str, required=False)
    parser.add_argument('--cache_path', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # skip experiment if seed in ignore list
    taskplan.utilities.utils.check_skip_protocol(args)
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    evaluate_main(args)
