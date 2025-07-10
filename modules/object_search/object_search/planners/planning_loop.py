import time
from common import Pose


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
            print(str(Pose(*chosen_subgoal_position[:2])))
            print(str(chosen_subgoal_position[:2]))
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
        print(f"Searching in container: {self.graph.get_node_name_by_idx(self.chosen_subgoal)}")
