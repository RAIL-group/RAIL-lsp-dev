import time


class PlanningLoop():
    def __init__(self, target_obj_info, simulator, robot, args,
                 destination=None, verbose=False, close_loop=False):
        self.target_obj_info = target_obj_info
        self.simulator = simulator
        self.graph, self.grid, self.subgoals = self.simulator.initialize_graph_map_and_subgoals()
        self.goal = target_obj_info['container_idx']
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
        while (self.chosen_subgoal not in self.goal):

            if self.verbose:
                target_obj_name = self.target_obj_info['name']
                print(f"Need (WHAT): {target_obj_name}")
                goals = [self.graph.get_node_name_by_idx(goal) for goal in self.goal]
                print(f"From (WHERE): {goals}")

                print(f"Counter: {counter} | Count since last turnaround: "
                      f"{count_since_last_turnaround}")

            yield {
                'observed_graph': self.graph,
                'observed_grid': self.grid,
                'subgoals': self.subgoals,
                'robot_pose': self.robot.pose
            }

            self.graph, self.grid, self.subgoals = self.simulator.update_graph_map_and_subgoals(
                observed_graph=self.graph,
                observed_grid=self.grid,
                subgoals=self.subgoals,
                chosen_subgoal=self.chosen_subgoal
            )

            # update robot_pose with current action pose for next iteration of action
            self.robot.move(self.chosen_subgoal)

            counter += 1
            count_since_last_turnaround += 1
            if self.verbose:
                print("")

        # add initial robot pose at the end to close the search loop
        if self.close_loop:
            self.robot.all_poses.append(self.robot.all_poses[0])
        elif self.destination is not None and self.robot.all_poses[-1] != self.destination:
            self.robot.all_poses.append(self.destination)

        if self.verbose:
            print("TOTAL TIME:", time.time() - fn_start_time)

    def set_chosen_subgoal(self, new_chosen_subgoal):
        self.chosen_subgoal = new_chosen_subgoal
        print(f"Searching (WHERE): {self.graph.get_node_name_by_idx(self.chosen_subgoal)}")
