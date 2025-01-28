from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

import mr_task
import mr_task.toy_environment as env
from common import Pose


def _setup(args):
    map = env.ToyMap(seed=args.seed)
    coords_locations, location_object = map.coords_locations, map.location_objects
    start_pose = Pose(0, 0)
    robot_team = [mr_task.robot.Robot(start_pose) for _ in range(args.num_robots)]
    print(f'{location_object=}')
    objects = map.objects_in_environment
    print(f'{objects=}')
    specification = mr_task.specification.get_random_specification(objects=objects, seed=args.seed)

    if args.planner == 'learned':
        mrtask_planner = mr_task.planner.LearnedMRTaskPlanner(specification)
    elif args.planner == 'optimistic':
        mrtask_planner = mr_task.planner.OptimisticMRTaskPlanner(specification)

    planning_loop = mr_task.planner.MRTaskPlanningLoop(robot_team,
                                                       coords_locations,
                                                       location_object,
                                                       get_distance_to_node)
    # Planning Loop
    for step_data in planning_loop:
        mrtask_planner.update(
            step_data['robot_poses'],
            step_data['object_found'],
            step_data['container_nodes']
        )

        if mrtask_planner.dfa_planner.is_accepting_state(mrtask_planner.dfa_planner.state):
            print("Task Completed !!!")
            break

        joint_action, cost = mrtask_planner.compute_joint_action()
        planning_loop.update_joint_action(joint_action)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim([0, env.GRID_SIZE])
    ax.set_ylim([0, env.GRID_SIZE])
    for robot in robot_team:
        ax.plot([pose.x for pose in robot.all_poses], [pose.y for pose in robot.all_poses], 'x-', alpha=0.8)
    for coords, loc in coords_locations.items():
        ax.scatter(coords[0], coords[1], color='black', marker='o')
        ax.text(coords[0]+0.5, coords[1]+0.5, loc)
    plt.title(f'{args.planner} | cost={robot_team[0].net_motion:.2f} | spec={specification}')

    imagename = Path(args.save_dir) / f'mtask_eval_planner_{args.planner}_seed_{args.seed}.png'
    plt.savefig(imagename)
    print(f"Cost = {robot_team[0].net_motion}")

    logfile = Path(args.save_dir) / f'log_{args.num_robots}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | COST : {robot_team[0].net_motion:0.3f}\n")



def get_distance_to_node(robot, node):
    return np.linalg.norm(np.array((robot.pose.x, robot.pose.y)) - np.array(node.location))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/mr_task/mr_task_eval')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='learned')
    parser.add_argument('--num_robots', type=int, default=2)
    args = parser.parse_args()
    _setup(args)
