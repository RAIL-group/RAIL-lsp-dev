import time
import numpy as np
import gridmap
import lsp
from mrlsp.utils.utility import find_robots_within_range, robot_team_communicate_data


class MRPlanningLoop():
    def __init__(self, goals, known_map, simulator, unity_bridge, robots, args, verbose=True):
        self.num_robots = len(robots)
        self.goals = goals
        self.known_map = known_map
        self.simulator = simulator
        self.unity_bridge = unity_bridge
        # instead of robot, we have robots
        self.robots = robots
        self.args = args
        self.is_stuck = [False for _ in range(self.num_robots)]
        self.verbose = verbose
        # instead of one chosen_subgoal, we have multiple chosen_subgoals
        self.chosen_subgoals = [None for _ in range(self.num_robots)]

        ''' Other variables'''
        self.goal_reached = None
        self.paths = [None for _ in range(self.num_robots)]
        self.timestamp = 0  # This keeps track of timestamp of data

    def _goal_reached(self):
        self.goal_reached = [not (np.abs(robot.pose.x - goal.x) >= 3 *
                             self.args.step_size or np.abs(robot.pose.y - goal.y)
                             >= 3 * self.args.step_size) for robot, goal in zip(self.robots, self.goals)]
        return any(self.goal_reached)

    def __iter__(self):
        counter = 0
        count_since_last_turnaround = [100 for _ in range(self.num_robots)]
        fn_start_time = time.time()
        # instead of one robot_grid, we have multiple robot_grids
        robot_grids = [lsp.constants.UNOBSERVED_VAL * np.ones_like(self.known_map) for _ in range(self.num_robots)]

        # Main planning loop (We can stop if any of the robot reaches the goal)
        while not self._goal_reached():

            if self.verbose:
                print(f"Goal: {self.goals[0].x}, {self.goals[0].y}")
                for i, robot in enumerate(self.robots):
                    print(f"Robot {i}: ({robot.pose.x:.2f}, {robot.pose.y:.2f}) [motion: {robot.net_motion:.2f}]")
                    print(f"Counter: {counter} | Count since last turnaround: "
                          f"{count_since_last_turnaround[i]}")
                print("--------------------------------------------------")

            # compute observations and update map
            pano_images = [self.simulator.get_image(robot, do_get_segmentation=True)[0] for robot in self.robots]
            _, robot_grids, visible_regions = zip(
                *[self.simulator.get_laser_scan_and_update_map(robot, robot_grid, True) for robot, robot_grid in zip(
                    self.robots, robot_grids)])

            # find the robot teams that are within communication range
            robots_within_range = find_robots_within_range(self.robots, self.args.comm_range)

            # update the robot data according to the robot teams
            robot_team_communicate_data(robots_within_range, self.robots, robot_grids)

            robot_grids = [self.robots[i].global_occupancy_grid for i in range(self.num_robots)]

            # compute intermediate map grids for planning
            visibility_masks = [gridmap.utils.inflate_grid(visible_region, 1.8, -0.1, 1.0)
                                for visible_region in visible_regions]
            '''NOTE: The computation here can be saved by finding inflated grid for
            one robot in a team, and assigning it for every robot in the same team'''
            inflated_grids = [self.simulator.get_inflated_grid(
                robot_grid, robot) for robot_grid, robot in zip(robot_grids, self.robots)]
            inflated_grids = [gridmap.mapping.get_fully_connected_observed_grid(
                inflated_grid, robot.pose) for inflated_grid, robot in zip(inflated_grids, self.robots)]

            # compute the subgoals
            '''NOTE: The computation here can be saved by finding subgoals for one
            robot in a team, and assigning it for every robot in the same team'''
            subgoals = [self.simulator.get_updated_frontier_set(
                inflated_grid, robot, set()) for inflated_grid, robot in zip(inflated_grids, self.robots)]

            # return the observation
            yield {
                'subgoals': subgoals,
                'images': pano_images,
                'robot_grids': robot_grids,
                'robot_poses': [robot.pose for robot in self.robots],
                'visibility_masks': visibility_masks,
                'robots_within_range': robots_within_range,
            }

            # compute the planning grid for each robot
            # If the chosen subgoal is None, plan via Dijkstra planner
            planning_grids = []
            for i in range(self.num_robots):
                if self.chosen_subgoals[i] is None:
                    planning_grids.append(lsp.core.mask_grid_with_frontiers(inflated_grids[i], [],))
                else:
                    planning_grids.append(lsp.core.mask_grid_with_frontiers(
                        inflated_grids[i], subgoals[i], do_not_mask=self.chosen_subgoals[i],))

            # check that the plan is feasible and compute path
            cost_grids = []
            paths = []
            for i in range(self.num_robots):
                cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                    planning_grids[i], [self.goals[i].x, self.goals[i].y], use_soft_cost=True)
                did_plan, path = get_path([self.robots[i].pose.x, self.robots[i].pose.y],
                                          do_sparsify=True, do_flip=True)

                cost_grids.append(cost_grid)
                paths.append(path)
            self.paths = paths

            for i in range(self.num_robots):
                # Move the robot
                '''The robot has to carry some data when it's moving so that it can send those data
                to other robots that come within its communication range.'''
                data = {
                    'subgoal': self.chosen_subgoals[i],
                    'timestamp': self.timestamp,
                }
                motion_primitives = self.robots[i].get_motion_primitives()
                do_use_path = (count_since_last_turnaround[i] > 10)
                costs, _ = lsp.primitive.get_motion_primitive_costs(
                    planning_grids[i],
                    cost_grids[i],
                    self.robots[i].pose,
                    paths[i],
                    motion_primitives,
                    do_use_path=do_use_path)

                if abs(min(costs)) < 1e10:
                    primitive_ind = np.argmin(costs)
                    self.robots[i].move(motion_primitives, primitive_ind, data)
                    if primitive_ind == len(motion_primitives) - 1:
                        count_since_last_turnaround[i] = -1
                else:
                    # Force the robot to return to known space
                    cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                        planning_grids[i], [self.goals[i].x, self.goals[i].y],
                        use_soft_cost=True,
                        obstacle_cost=1e5)
                    did_plan, path = get_path([self.robots[i].pose.x, self.robots[i].pose.y],
                                              do_sparsify=True,
                                              do_flip=True)
                    cost_grids[i] = cost_grid
                    paths[i] = path
                    self.paths[i] = path
                    # add self.paths
                    costs, _ = lsp.primitive.get_motion_primitive_costs(
                        planning_grids[i],
                        cost_grids[i],
                        self.robots[i].pose,
                        paths[i],
                        motion_primitives,
                        do_use_path=False)
                    self.robots[i].move(motion_primitives, np.argmin(costs), data)

                # Check that the robot is not 'stuck'.
                if self.robots[i].max_travel_distance(
                        num_recent_poses=100) < 5 * self.args.step_size:
                    print(f"Planner for robot {i} stuck")
                    self.is_stuck[i] = True
                    if np.all(self.is_stuck):
                        print("All robots stuck. Exiting.")
                        return
                    continue

                if self.robots[i].net_motion > 4000:
                    print(f"Robot {i} reached maximum distance.")
                    self.is_stuck[i] = True
                    if np.all(self.is_stuck):
                        print("All robots (length > 4000). Exiting.")
                        return
                    continue

                count_since_last_turnaround[i] += 1

            counter += 1

        if self.verbose:
            print("TOTAL TIME:", time.time() - fn_start_time)
        print(f"Planning loop iteration complete, {self.goal_reached}")

    def set_chosen_subgoals(self, new_chosen_subgoals, timestamp=None):
        self.chosen_subgoals = new_chosen_subgoals
        self.timestamp = timestamp
