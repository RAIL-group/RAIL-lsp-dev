import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import procthor
import object_search

from object_search.planners import (
    LSPLLMLlamaPlanner,
    FullLLMLlamaPlanner,
    LSPLLMGPTOSSPlanner,
    FullLLMGPTOSSPlanner,
)
from object_search_select.planners import PolicySelectionPlanner
from procthor.simulators import SceneGraphSimulator
from pathlib import Path


def eval_main(args):
    thor_interface = procthor.ThorInterface(args=args)
    known_graph, known_grid, robot_pose, target_obj_info = thor_interface.gen_map_and_poses()

    simulator = SceneGraphSimulator(known_graph,
                                    args,
                                    target_obj_info,
                                    known_grid,
                                    thor_interface)

    robot = object_search.robot.Robot(robot_pose)
    planners = [
        # LSPLLMLlamaPlanner(target_obj_info, args, prompt_template_id='prompt_a'),
        # LSPLLMLlamaPlanner(target_obj_info, args, prompt_template_id='prompt_b'),
        # LSPLLMLlamaPlanner(target_obj_info, args, prompt_template_id='prompt_minimal'),
        # FullLLMLlamaPlanner(target_obj_info, args, prompt_template_id='prompt_direct'),
        LSPLLMGPTOSSPlanner(target_obj_info, args, prompt_template_id='prompt_a'),
        LSPLLMGPTOSSPlanner(target_obj_info, args, prompt_template_id='prompt_b'),
        LSPLLMGPTOSSPlanner(target_obj_info, args, prompt_template_id='prompt_minimal'),
        FullLLMGPTOSSPlanner(target_obj_info, args, prompt_template_id='prompt_direct'),
    ]
    chosen_planner_idx = args.planner_names.index(args.chosen_planner)
    planner = PolicySelectionPlanner(target_obj_info, planners, chosen_planner_idx, args)

    planning_loop = object_search.planners.planning_loop.PlanningLoop(
        target_obj_info, simulator, robot=robot, args=args,
        verbose=True)

    for counter, step_data in enumerate(planning_loop):
        planner.update(
            step_data['observed_graph'],
            step_data['observed_grid'],
            step_data['subgoals'],
            step_data['robot_pose'])

        chosen_subgoal = planner.compute_selected_subgoal()
        planning_loop.set_chosen_subgoal(chosen_subgoal)

    cost, trajectory = object_search.utils.compute_cost_and_trajectory(known_grid, robot.all_poses)

    plt.figure(figsize=(8, 8))
    known_locations = [known_graph.get_node_name_by_idx(idx) for idx in target_obj_info['container_idxs']]
    plt.suptitle(f"Seed: {args.current_seed} | Target object: {target_obj_info['name']}\n"
                 f"Known locations: {known_locations} ")

    ax = plt.subplot(221)
    plt.title('Whole scene graph')
    procthor.plotting.plot_graph(ax, known_graph.nodes, known_graph.edges)

    ax = plt.subplot(222)
    procthor.plotting.plot_graph_on_grid(ax, known_grid, known_graph)
    plt.text(robot_pose.x, robot_pose.y, '+', color='red', size=6, rotation=45)
    plt.title('Graph over occupancy grid')

    plt.subplot(223)
    top_down_image = thor_interface.get_top_down_image()
    plt.imshow(top_down_image)
    plt.title('Top-down view of the map')
    plt.axis('off')

    plt.subplot(224)
    ax = plt.subplot(224)
    object_search.plotting.plot_grid_with_robot_trajectory(ax, known_grid, robot.all_poses, trajectory, known_graph)
    # object_search.plotting.plot_grid_with_robot_trajectory_gradient(ax,
    #                                                                 known_grid,
    #                                                                 robot.all_poses,
    #                                                                 trajectory,
    #                                                                 known_graph,
    #                                                                 cmap=args.planner_cmaps[args.chosen_planner])
    plt.title(f"Planner: {args.chosen_planner} Cost: {cost:0.2f}")

    plt.savefig(Path(args.save_dir) /
                f'img_plcy_{args.chosen_planner}_envrnmnt_{args.env}_{args.current_seed}.png', dpi=1000)
    # plt.savefig(Path(args.save_dir) /
    #             f'img_plcy_{args.chosen_planner}_envrnmnt_{args.env}_{args.current_seed}.pdf', dpi=1000)
    return cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation for Object Search")
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--do_not_replay', action='store_true')
    parser.add_argument('--do_plot', action='store_true')
    planner_names = [
        # 'lspllamaprompta', 'lspllamapromptb', 'lspllamapromptminimal',
        # 'fullllamapromptdirect',
        'lspgptossprompta', 'lspgptosspromptb', 'lspgptosspromptminimal',
        'fullgptosspromptdirect'
    ]
    parser.add_argument('--chosen_planner', choices=planner_names)
    parser.add_argument('--env', choices=['apartment'])
    args = parser.parse_args()

    random.seed(args.current_seed)
    np.random.seed(args.current_seed)

    args.planner_names = planner_names

    path = Path(args.save_dir)
    target_file = path / f'target_plcy_{args.chosen_planner}_envrnmnt_{args.env}_{args.current_seed}.txt'

    if target_file.is_file():
        print(f'Data already exists for {args.chosen_planner}_{args.env}_{args.current_seed}.')
        exit()

    print(f'Generating data for {args.chosen_planner}_{args.env}_{args.current_seed}.')

    cost = eval_main(args)

    with open(target_file, 'w') as f:
        f.write(f"{cost}\n")
