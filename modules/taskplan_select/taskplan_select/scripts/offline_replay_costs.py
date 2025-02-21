import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import procthor
import taskplan
# from taskplan.planners import (
#     ClosestActionPlanner,
#     LearnedPlanner,
#     KnownPlanner
# )
from taskplan_select.planners import (LSPLLMGPT4Planner,
                                      LSPLLMGeminiPlanner,
                                      FullLLMGPT4Planner,
                                      FullLLMGeminiPlanner,
                                      PolicySelectionPlanner)
from taskplan_select.simulators import SceneGraphSimulator
from pathlib import Path


def eval_main(args):
    thor_interface = procthor.ThorInterface(args=args)
    known_graph, known_grid, robot_pose, target_obj_info = thor_interface.gen_map_and_poses()

    simulator = SceneGraphSimulator(known_graph,
                                    args,
                                    target_obj_info,
                                    known_grid,
                                    thor_interface)

    robot = taskplan.robot.Robot(robot_pose)
    planners = [
        LSPLLMGPT4Planner(target_obj_info, args, fake_llm_response_text='100%'),
        LSPLLMGPT4Planner(target_obj_info, args, prompt_template_id=0),
        # LSPLLMGPT4Planner(target_obj_info, args, prompt_template_id=1),
        LSPLLMGPT4Planner(target_obj_info, args, prompt_template_id=2),
        LSPLLMGPT4Planner(target_obj_info, args, prompt_template_id=6),
        LSPLLMGeminiPlanner(target_obj_info, args, prompt_template_id=0),
        # LSPLLMGeminiPlanner(target_obj_info, args, prompt_template_id=1),
        LSPLLMGeminiPlanner(target_obj_info, args, prompt_template_id=2),
        LSPLLMGeminiPlanner(target_obj_info, args, prompt_template_id=6),
        FullLLMGPT4Planner(target_obj_info, args, prompt_template_id=0),
        FullLLMGeminiPlanner(target_obj_info, args, prompt_template_id=0)
    ]
    chosen_planner_idx = args.planner_names.index(args.chosen_planner)
    planner = PolicySelectionPlanner(target_obj_info, planners, chosen_planner_idx, args)

    planning_loop = taskplan.planners.planning_loop.PlanningLoop(
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

    path = robot.all_poses
    cost, trajectory = taskplan.core.compute_path_cost(known_grid, path)

    plt.figure(figsize=(8, 8))
    what = target_obj_info['name']
    where = [known_graph.get_node_name_by_idx(goal) for goal in target_obj_info['container_idx']]
    plt.suptitle(f"Find {what} from {where} in seed: [{args.current_seed}]")

    ax = plt.subplot(221)
    plt.title('Whole scene graph')
    procthor.plotting.plot_graph(ax, known_graph.nodes, known_graph.edges)
    plt.axis('off')

    plt.subplot(222)
    procthor.plotting.plot_graph_on_grid(known_grid, known_graph)
    x, y = robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title('Graph over occupancy grid')

    plt.subplot(223)
    top_down_frame = simulator.get_top_down_image_orthographic()
    plt.imshow(top_down_frame)
    plt.title('Top-down view of the map')
    plt.axis('off')

    plt.subplot(224)
    plotting_grid = procthor.plotting.make_plotting_grid(np.transpose(known_grid))
    plt.imshow(plotting_grid)
    plt.title(f"Planner: {args.chosen_planner} Cost: {cost:0.3f}")
    x, y = path[0]
    plt.text(x, y, '0 - ROBOT', color='brown', size=4)

    for idx, coords in enumerate(path[1:]):
        pose = taskplan.utilities.utils. \
            get_pose_from_coord(coords, known_graph.nodes)
        name = known_graph.get_node_name_by_idx(pose)
        x, y = coords
        plt.text(x, y, f'{idx+1} - {pose}:{name}', color='brown', size=4)

    viridis_cmap = plt.get_cmap('viridis')
    colors = np.linspace(0, 1, len(trajectory[0]))
    line_colors = viridis_cmap(colors)
    plt.plot(trajectory[0], trajectory[1])
    # for idx, x in enumerate(trajectory[0]):
    #     y = trajectory[1][idx]
    #     # plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)

    plt.savefig(Path(args.save_dir) /
                f'img_plcy_{args.chosen_planner}_envrnmnt_{args.env}_{args.current_seed}.png', dpi=1000)
    plt.savefig(Path(args.save_dir) /
                f'img_plcy_{args.chosen_planner}_envrnmnt_{args.env}_{args.current_seed}.pdf', dpi=1000)
    return planner, robot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation for Object Search")
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--do_not_replay', action='store_true')
    # parser.add_argument('--network_file', type=str)
    parser.add_argument('--do_plot', action='store_true')
    planner_names = ['prompttrivial',
                     'lspgptpromptone', 'lspgptprompttwo', 'lspgptpromptthree',  # 'lspgptpromptfour',
                     'lspgeminipromptone', 'lspgeminiprompttwo', 'lspgeminipromptthree',  # 'lspgeminipromptfour',
                     'fullgptpromptone',
                     'fullgeminipromptone']
    parser.add_argument('--chosen_planner', choices=planner_names)
    parser.add_argument('--env', choices=['apartment'])
    args = parser.parse_args()

    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    # torch.manual_seed(args.current_seed)

    args.planner_names = planner_names
    all_planners = '_'.join(args.planner_names)

    path = Path(args.save_dir)
    cost_file = path / f'cost_{args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.txt'
    lb_costs_file = path / f'lbc_{args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.txt'
    target_file = path / f'target_plcy_{args.chosen_planner}_envrnmnt_{args.env}_{args.current_seed}.txt'

    if cost_file.is_file():
        print(f'Data already exists for {args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.')
        exit()

    print(f'Generating data for {args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.')

    planner, robot = eval_main(args)

    if args.do_not_replay:
        exit()

    costs, lb_costs = planner.get_costs(robot)

    with open(cost_file, 'w') as f:
        np.savetxt(f, costs)
    with open(lb_costs_file, 'w') as f:
        np.savetxt(f, lb_costs)
    with open(target_file, 'w') as f:
        f.write('\n')
