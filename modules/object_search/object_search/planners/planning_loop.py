import time
import numpy as np
from common import Pose
import lsp
import gridmap
import object_search


class PlanningLoop():
    def __init__(self, target_obj_info, simulator, robot, args,
                 destination=None, verbose=True, close_loop=False):
        self.target_obj_info = target_obj_info
        self.simulator = simulator
        self.graph, self.grid, self.subgoals = self.simulator.initialize_graph_grid_and_containers()
        self.goal_containers = target_obj_info['container_idxs']
        self.robot = robot
        self.destination = destination
        self.args = args
        self.did_succeed = True
        self.verbose = verbose
        self.chosen_subgoal = None
        self.close_loop = close_loop

    def __iter__(self):
        counter = 0
        count_since_last_turnaround = 100
        fn_start_time = time.time()

        # Main planning loop
        while (self.chosen_subgoal not in self.goal_containers):

            if self.verbose:
                target_obj_name = self.target_obj_info['name']
                print(f"Target object: {target_obj_name}")
                print(f"Known locations: {[self.graph.get_node_name_by_idx(idx) for idx in self.goal_containers]}")

                print(f"Counter: {counter} | Count since last turnaround: "
                      f"{count_since_last_turnaround}")

            yield {
                'observed_graph': self.graph,
                'observed_grid': self.grid,
                'subgoals': self.subgoals,
                'robot_pose': self.robot.pose
            }

            self.graph, self.grid, self.subgoals = self.simulator.update_graph_grid_and_containers(
                self.graph,
                self.grid,
                self.subgoals,
                self.chosen_subgoal
            )

            # Move robot to chosen subgoal pose
            chosen_subgoal_position = self.graph.get_node_position_by_idx(self.chosen_subgoal)
            self.robot.move(Pose(*chosen_subgoal_position[:2]))

            counter += 1
            count_since_last_turnaround += 1
            if self.verbose:
                print(" ")

        # Add initial robot pose at the end to close the search loop
        if self.close_loop:
            self.robot.all_poses.append(self.robot.all_poses[0])
        elif self.destination is not None and self.robot.all_poses[-1] != self.destination:
            self.robot.all_poses.append(self.destination)

        if self.verbose:
            print("TOTAL TIME: ", time.time() - fn_start_time)

    def set_chosen_subgoal(self, new_chosen_subgoal):
        self.chosen_subgoal = new_chosen_subgoal


class PlanningLoopPartialGrid():
    def __init__(self, target_obj_info, simulator, robot, args,
                 destination=None, verbose=True, close_loop=False):
        self.target_obj_info = target_obj_info
        self.simulator = simulator
        self.goal_containers = target_obj_info['container_idxs']
        self.robot = robot
        self.destination = destination
        self.args = args
        self.did_succeed = True
        self.verbose = verbose
        self.chosen_subgoal = None
        self.close_loop = close_loop
        self.current_path = None
        self.newly_found_objects = set()

    def is_goal_reached(self):
        if self.target_obj_info['name'] in self.newly_found_objects:
            return True
        return False

    def __iter__(self):
        counter = 0
        count_since_last_turnaround = 100
        self.observed_graph = self.simulator.known_graph.get_room_only_graph()
        self.observed_grid = lsp.constants.UNOBSERVED_VAL * np.ones_like(self.simulator.known_grid)
        fn_start_time = time.time()

        # Main planning loop
        while not self.is_goal_reached():
            if self.verbose:
                target_obj_name = self.target_obj_info['name']
                print(f"Target object: {target_obj_name}")
                known_locations = [self.simulator.known_graph.get_node_name_by_idx(idx) for idx in self.goal_containers]
                print(f"Known locations: {known_locations}")

                print(f"Counter: {counter} | Count since last turnaround: "
                      f"{count_since_last_turnaround}")

            simulator_data = self.simulator.update_graph_and_map(self.robot, self.observed_graph, self.observed_grid)

            self.observed_graph = simulator_data['observed_graph']
            self.observed_grid = simulator_data['observed_grid']
            self.newly_found_objects = simulator_data['newly_found_objects']
            inflated_grid = simulator_data['inflated_grid']
            frontiers = simulator_data['frontiers']

            yield {
                'observed_graph': self.observed_graph,
                'observed_grid': self.observed_grid,
                'inflated_grid': inflated_grid,
                'containers': simulator_data['containers'],
                'frontiers': frontiers,
                'robot_pose': self.robot.pose,
                'ranges': simulator_data['ranges'],
                'visible_mask': simulator_data['visibility_mask'],
            }

            if isinstance(self.chosen_subgoal, object_search.core.Subgoal):  # chosen_subgoal is a container
                planning_grid = lsp.core.mask_grid_with_frontiers(
                    inflated_grid, frontiers)
                chosen_subgoal_position = self.observed_graph.get_node_position_by_idx(self.chosen_subgoal.id)
                print(f"Searching in container: {self.observed_graph.get_node_name_by_idx(self.chosen_subgoal.id)}")
            else:  # chosen_subgoal is a frontier
                planning_grid = lsp.core.mask_grid_with_frontiers(
                    inflated_grid, frontiers, do_not_mask=self.chosen_subgoal)
                chosen_subgoal_position = self.chosen_subgoal.get_frontier_point()
                print(f"Exploring frontier at {chosen_subgoal_position}")

            # Check that the plan is feasible and compute path
            cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                planning_grid, chosen_subgoal_position, use_soft_cost=True)
            did_plan, path = get_path([self.robot.pose.x, self.robot.pose.y],
                                      do_sparsify=True, do_flip=True)
            self.current_path = path

            # Move the robot
            motion_primitives = self.robot.get_motion_primitives()
            do_use_path = (count_since_last_turnaround > 10)
            costs, _ = lsp.primitive.get_motion_primitive_costs(
                planning_grid,
                cost_grid,
                self.robot.pose,
                path,
                motion_primitives,
                do_use_path=do_use_path)
            if abs(min(costs)) < 1e10:
                primitive_ind = np.argmin(costs)
                self.robot.move(motion_primitives, primitive_ind)
                if primitive_ind == len(motion_primitives) - 1:
                    count_since_last_turnaround = -1
            else:
                # Force the robot to return to known space
                cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                    planning_grid, chosen_subgoal_position,
                    use_soft_cost=True,
                    obstacle_cost=1e5)
                did_plan, path = get_path([self.robot.pose.x, self.robot.pose.y],
                                          do_sparsify=True,
                                          do_flip=True)
                self.current_path = path
                costs, _ = lsp.primitive.get_motion_primitive_costs(
                    planning_grid,
                    cost_grid,
                    self.robot.pose,
                    path,
                    motion_primitives,
                    do_use_path=False)
                self.robot.move(motion_primitives, np.argmin(costs))

            # Check that the robot is not 'stuck'.
            if self.robot.max_travel_distance(
                    num_recent_poses=100) < 5 * self.args.step_size:
                print("Planner stuck")
                self.did_succeed = False
                break

            if self.robot.net_motion > 4000:
                print("Reached maximum distance.")
                self.did_succeed = False
                break

            if self.verbose:
                print(" ")

            counter += 1
            count_since_last_turnaround += 1

        if self.verbose:
            print("TOTAL TIME: ", time.time() - fn_start_time)

    def set_chosen_subgoal(self, new_chosen_subgoal):
        self.chosen_subgoal = new_chosen_subgoal
