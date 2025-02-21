import random
import argparse
import numpy as np
# import matplotlib.pyplot as plt
import procthor
# import taskplan

from taskplan_select.simulators import SceneGraphSimulator
from pathlib import Path
from PIL import Image


# def evaluate_main(args):
#     thor_interface = procthor.ThorInterface(args=args)
#     known_graph, known_grid, robot_pose, target_obj_info = thor_interface.gen_map_and_poses()

#     simulator = SceneGraphSimulator(known_graph,
#                                     args,
#                                     target_obj_info,
#                                     known_grid,
#                                     thor_interface)
#     image = simulator.get_top_down_image_orthographic()


#     robot = taskplan.robot.Robot(robot_pose)
#     planners = [LSPLLMPlanner(target_obj_info, args, prompt_template_id=0,
#                             #   fake_llm_response_text='100%'
#                               ),
#                 KnownPlanner(target_obj_info, args, known_graph, known_grid, args)]
#     planner = PolicySelectionPlanner(target_obj_info, planners, args.chosen_planner_idx, args)
#     planning_loop = taskplan.planners.planning_loop.PlanningLoop(
#         target_obj_info, simulator, robot=robot, args=args,
#         verbose=True)

#     for counter, step_data in enumerate(planning_loop):
#         # Update the planner objects
#         planner.update(
#             step_data['observed_graph'],
#             step_data['observed_grid'],
#             step_data['subgoals'],
#             step_data['robot_pose'])

#         # Compute the next subgoal and set to the planning loop
#         chosen_subgoal = planner.compute_selected_subgoal()
#         planning_loop.set_chosen_subgoal(chosen_subgoal)

#     # path = robot.all_poses
#     # dist, trajectory = taskplan.core.compute_path_cost(known_grid, path)

#     planner_costs, lb_costs = planner.get_costs(robot)
#     print(planner_costs, lb_costs)


def save_env_image(args):
    thor_interface = procthor.ThorInterface(args=args)
    known_graph, known_grid, robot_pose, target_obj_info = thor_interface.gen_map_and_poses()

    simulator = SceneGraphSimulator(known_graph,
                                    args,
                                    target_obj_info,
                                    known_grid,
                                    thor_interface)
    print(f'Getting top down image for seed {args.current_seed}')
    if args.orthographic:
        image = simulator.get_top_down_image_orthographic()
    else:
        image = simulator.get_top_down_image()
    save_dir = Path(args.save_dir)
    Image.fromarray(image).save(save_dir / f'env_image_{args.current_seed}.png')


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for Object Search"
    )
    parser.add_argument('--current_seed', type=int, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--resolution', type=float, required=True)
    # parser.add_argument('--network_file', type=str, required=False)
    parser.add_argument('--orthographic', action='store_true')
    # parser.add_argument('--chosen_planner_idx', type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    # torch.manual_seed(args.current_seed)
    # evaluate_main(args)
    save_env_image(args)
