import numpy as np
import potlp
import lsp
import gridmap


class BasePOTLPPlanner(object):
    def __init__(self, args, known_map, ltl_planner, all_nodes, robot, simulator, verbose=False, iterations=1000):
        self.args = args
        self.ltl_planner = ltl_planner
        self.dfa_state = self.ltl_planner.get_init_dfa_state()
        self.all_nodes = all_nodes
        self.robot = robot
        self.simulator = simulator
        self.observed_nodes = []
        self.subgoal_nodes = []
        self.node_path = None
        self.verbose = verbose
        self.known_map = known_map
        self.robot_grid =  lsp.constants.UNOBSERVED_VAL * np.ones_like(self.known_map)
        self.counter = 0
        self.iterations = iterations
        self.ordered_nodes = list(self.ltl_planner.semantic_index.keys())
        self.known_space_nodes_ordered = []

        for node_name in ltl_planner.semantic_index.keys():
            for node in self.all_nodes:
                if node.props[0] == node_name:
                    self.known_space_nodes_ordered.append(node)
    
    def get_node_path(self, observed_nodes, subgoal_nodes):
        subgoal_prop_dict = self.get_subgoal_props(subgoal_nodes)
        travel_cost_dict =  potlp.core.get_travel_cost_dict(
                                self.inflated_grid,
                                observed_nodes + subgoal_nodes,
                                self.robot.pose)
        action_dict, node_id_dict = self.ltl_planner._get_actions(
            observed_nodes, subgoal_nodes, travel_cost_dict)
        # If everything is in known space, don't call potlp, just complete the specification
        selected_action = potlp.find_best_action_accel(
            (node_id_dict["robot"], self.dfa_state), 
            action_dict, 
            subgoal_prop_dict, 
            node_id_dict,
            self.iterations)
        print(f'Action: {selected_action}')
        node_path = selected_action.to_node_path(node_id_dict)
        print(f'Node path: {node_path}')
        return node_path

    def get_subgoal_props(self, subgoal_nodes):
        raise NotImplementedError

    def step(self):
        if self.verbose:
            print(f"DFA state: {self.dfa_state}")
            print(f"Robot: {self.robot.pose.x}, {self.robot.pose.y} [motion: {self.robot.net_motion}]")
            print(f"Counter: {self.counter}")
        # Compute observations and update map, observed nodes
        _, self.robot_grid = self.simulator.get_laser_scan_and_update_map(
            self.robot, self.robot_grid)
        observed_nodes = [ks_node for ks_node in self.known_space_nodes_ordered 
                        if self.robot_grid[ks_node.position[0], ks_node.position[1]] == 
                        lsp.constants.FREE_VAL]

        # Compute intermediate map grids for planning
        self.inflated_grid = self.simulator.get_inflated_grid(self.robot_grid, self.robot)
        self.inflated_grid = gridmap.mapping.get_fully_connected_observed_grid(
            self.inflated_grid, self.robot.pose)
        
        # Update frontiers
        self.frontiers = self.simulator.get_updated_frontier_set(
            self.inflated_grid, self.robot, set())
        subgoal_nodes = [potlp.core.Node(is_subgoal=True, position=f.get_frontier_point(), subgoal=f) 
                        for f in self.frontiers]

        # Get the path to the next node based on which is feasible
        # TODO: Abhish: Change function name;
        self.node_path = self.get_node_path(observed_nodes, subgoal_nodes)
        
        waypoint = self.node_path[1].position
        chosen_subgoal = self.node_path[1].subgoal
        self.planning_grid = lsp.core.mask_grid_with_frontiers(
            self.inflated_grid, self.frontiers, do_not_mask=chosen_subgoal)
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                self.planning_grid, waypoint, use_soft_cost=True)
        did_plan, path = get_path([self.robot.pose.x, self.robot.pose.y],
                                do_sparsify=True, do_flip=True)
        # Get the motion primitives: primitive motions the robot can execute
        motion_primitives = self.robot.get_motion_primitives()
        
        # Pick the "min cost" primitive, which best moves along the path
        costs, _ = lsp.primitive.get_motion_primitive_costs(
                    self.inflated_grid, cost_grid, self.robot.pose, path,
                    motion_primitives, do_use_path=True)
        best_primitive_index = np.argmin(costs)
        
        # Move according to that primitive action
        self.robot.move(motion_primitives, np.argmin(costs))
        
        # Update the DFA state
        nearby_nodes = [node for node in observed_nodes
                    if ((self.robot.pose.x - node.position[0]) ** 2 + 
                        (self.robot.pose.y - node.position[1]) ** 2) < 4.0]

        for node in nearby_nodes:
            self.dfa_state = self.ltl_planner.update_dfa_state(self.dfa_state, node)

        # Check that the robot is not 'stuck'.
        if self.robot.max_travel_distance(
                num_recent_poses=100) < 5 * self.args.step_size:
            print("Planner stuck")
            return

        if self.robot.net_motion > 4000:
            print("Reached maximum distance.")
            return
        
        self.counter += 1
