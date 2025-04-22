import matplotlib.pyplot as plt
import gridmap
from gridmap import utils
from gridmap.constants import UNOBSERVED_VAL, FREE_VAL, COLLISION_VAL
from lsp_select.offline_replay import OfflineReplay
import lsp_select
import mrlsp
from pathlib import Path


class MultiRobotOfflineReplay(OfflineReplay):
    def __init__(self, partial_map, poses, images, nearest_pose_data, final_subgoals, goal, args):
        super().__init__(partial_map, poses, images, nearest_pose_data, final_subgoals, goal, args)
        self.next_subgoals_to_mask_multi_robot = []

    def get_updated_frontier_set(self, inflated_grid, robot, saved_frontiers):
        """Compute the frontiers, store the new ones and compute properties."""
        frontiers = super().get_updated_frontier_set(inflated_grid, robot, saved_frontiers)
        self.next_subgoals_to_mask_multi_robot.append(self.next_subgoals_to_mask)
        # print(f'{len(self.next_subgoals_to_mask_multi_robot)=}')
        return frontiers


def get_lowerbound_planner_costs(navigation_data, planner, args):
    """Helper function to get optimistic and simply-connected lower bound cost through offline replay
    of a planner based on collected navigation data.
    """
    start_poses = [navigation_data['start'] for _ in range(args.num_robots)]
    goal_poses = [navigation_data['goal'] for _ in range(args.num_robots)]
    goal = goal_poses[0]
    partial_map = navigation_data['partial_map']
    final_subgoals = navigation_data['final_subgoals']

    robot_team = mrlsp.utils.utility.get_robot_team(num_robots=args.num_robots,
                                                    start_poses=start_poses,
                                                    primitive_length=args.step_size,
                                                    num_primitives=args.num_primitives,
                                                    map_data=None)
    simulator = MultiRobotOfflineReplay(partial_map,
                                        poses=navigation_data['poses'],
                                        images=navigation_data['images'],
                                        nearest_pose_data=navigation_data['nearest_pose_data'],
                                        final_subgoals=navigation_data['final_subgoals'],
                                        goal=goal,
                                        args=args)
    simulator.frontier_grouping_inflation_radius = simulator.inflation_radius
    planning_loop = mrlsp.planners.MRPlanningLoop(goal_poses,
                                                  partial_map,
                                                  simulator,
                                                  unity_bridge=None,
                                                  robots=robot_team,
                                                  args=args)

    masked_frontiers = set()
    all_alt_costs = []

    for counter, step_data in enumerate(planning_loop):
        images = [im for im, _ in step_data['images']]
        nearest_poses = [p for _, p in step_data['images']]
        planner.update(
            {'images': images},
            step_data['robot_grids'],
            step_data['subgoals'],
            step_data['robot_poses'],
            step_data['visibility_masks'])
        planning_loop.set_chosen_subgoals(planner.compute_selected_subgoal())

        inflated_grid = utils.inflate_grid(
            partial_map, inflation_radius=args.inflation_radius_m / args.base_resolution)

        frontier_grid = inflated_grid.copy()
        frontier_grid[inflated_grid == FREE_VAL] = COLLISION_VAL
        frontier_grid[inflated_grid == UNOBSERVED_VAL] = FREE_VAL
        goal_grid = inflated_grid.copy()
        goal_grid[inflated_grid == UNOBSERVED_VAL] = COLLISION_VAL

        # Open up all frontiers
        for f in final_subgoals:
            frontier_grid[f.points[0, :], f.points[1, :]] = FREE_VAL
            goal_grid[f.points[0, :], f.points[1, :]] = FREE_VAL

        for i in range(args.num_robots):
            simulator.next_subgoals_to_mask = simulator.next_subgoals_to_mask_multi_robot[i]
            # Block already masked frontiers in both grids
            masked_frontiers.update(simulator.next_subgoals_to_mask)
            for f in masked_frontiers:
                frontier_grid[f.points[0, :], f.points[1, :]] = COLLISION_VAL
                goal_grid[f.points[0, :], f.points[1, :]] = COLLISION_VAL

            # Block 'next mask' frontiers for goal grid
            for f in simulator.next_subgoals_to_mask:
                frontier_grid[f.points[0, :], f.points[1, :]] = FREE_VAL
                goal_grid[f.points[0, :], f.points[1, :]] = COLLISION_VAL

            cost_grid_goal, _ = gridmap.planning.compute_cost_grid_from_position(
                goal_grid, [goal.x, goal.y], use_soft_cost=True)

            alt_costs_to_goal = []
            for f in simulator.next_subgoals_to_mask:
                cost_grid_frontier, _ = gridmap.planning.compute_cost_grid_from_position(
                    frontier_grid, f.get_frontier_point(), use_soft_cost=True)

                total_cost_grid = cost_grid_frontier + cost_grid_goal
                costs_temp = []
                for frontier in final_subgoals:
                    if frontier in masked_frontiers:
                        continue
                    f_x, f_y = frontier.get_frontier_point()
                    costs_temp.append(total_cost_grid[f_x, f_y])
                if len(costs_temp) != 0:
                    alt_costs_to_goal.append(costs_temp)

            if len(alt_costs_to_goal) != 0:
                all_alt_costs.append([robot_team[i].net_motion, alt_costs_to_goal])
        simulator.next_subgoals_to_mask_multi_robot = []

        if args.do_plot:
            plt.ion()
            plt.figure(1)
            plt.clf()
            mrlsp.utils.plotting.visualize_robots(args,
                                                  robot_team,
                                                  planner.observed_map,
                                                  images,
                                                  planner.subgoals,
                                                  goal_poses[0],
                                                  planning_loop.paths,
                                                  simulator.known_map,
                                                  timestamp=None)
            alt_costs_min = []
            for net_motion, alt_costs in all_alt_costs:
                alt_costs_min.append(net_motion + min([min(c) if len(c) > 0 else float('inf')
                                                       for c in alt_costs]))

            lbopt = min(alt_costs_min) if len(alt_costs_min) != 0 else float('inf')
            plt.suptitle(f'Offline Replay with {args.planner_names[args.replayed_planner_idx]}\n'
                         r'$C^{lb,opt}$= ' f'{lbopt:.2f}, ' r'$C^{lb,s.c.}$= '
                         f'{max([r.net_motion for r in robot_team]):.2f}')

            for p in nearest_poses:
                plt.scatter(p[0], p[1], s=20, marker='s', edgecolors='black')
            ax = plt.subplot(1, 2, 1)
            lsp_select.utils.plotting.plot_grid_with_frontiers(
                ax, simulator.known_map, None, frontiers=[])
            lsp_select.utils.plotting.plot_pose_path(ax, navigation_data['robot_path'])
            save_dir = Path(args.save_dir) / 'replay_imgs'
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f'rep_{args.env}_{args.planner_names[args.chosen_planner_idx]}_'
                        f'{args.planner_names[args.replayed_planner_idx]}_{args.current_seed}_{counter}.png')
            plt.show()
            plt.pause(0.1)

        if max([r.net_motion for r in robot_team]) >= 3000:
            break

    if not any(planning_loop.goal_reached):
        net_motion = max([r.net_motion for r in robot_team])
    else:
        net_motion = robot_team[planning_loop.goal_reached.index(True)].net_motion

    # Compute optimistic lower bound
    alt_costs_min = []
    for net_motion, alt_costs in all_alt_costs:
        alt_costs_min.append(net_motion + min([min(c) if len(c) > 0 else float('inf')
                                               for c in alt_costs]))
    optimistic_lb = min(alt_costs_min) if len(alt_costs_min) != 0 else float('inf')
    simply_connected_lb = net_motion

    return optimistic_lb, simply_connected_lb
