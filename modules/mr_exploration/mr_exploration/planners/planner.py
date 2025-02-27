import numpy as np
import copy
import gridmap
import mrlsp
import lsp

class MRPlanner(object):
    def __init__(self, robots, args):
        self.name = 'MultirobotBasePlanner'
        self.args = args
        self.num_robots = len(robots)
        self.robots = robots
        self.observed_maps = None
        self.inflated_grids = None
        # Global observed map and inflated grid
        self.observed_map = None
        self.inflated_grid = None
        #self.goal_poses = goal_poses
        # same goal for all robots
        #self.goal = self.goal_poses[0]

    def update(self, observations, observed_maps, subgoals, robot_poses, visibility_masks):
        self.observations = observations
        self.observed_maps = observed_maps
        self.subgoals = subgoals[0]
        self.robot_poses = robot_poses
        self.visibility_masks = visibility_masks
        self.inflated_grids = self._get_inflated_occupancy_grids()
        # Since there's full communication, we can just use the first robot's observations as global
        self.inflated_grid = self.inflated_grids[0]
        self.observed_map = observed_maps[0]

    def compute_selected_subgoal(self):
        """Returns the selected subgoal (frontier)."""
        raise NotImplementedError()

    def _get_inflated_occupancy_grids(self):
        inflated_grids = []
        for i in range(self.num_robots):
            inflated_grid = gridmap.utils.inflate_grid(
                self.observed_maps[i], inflation_radius=self.inflation_radius)
            inflated_grid = gridmap.mapping.get_fully_connected_observed_grid(
                inflated_grid, self.robot_poses[i])
            inflated_grid[int(self.robot_poses[i].x), int(self.robot_poses[i].y)] = 0
            if np.isinf(self.args.comm_range):
                inflated_grids = [inflated_grid] * self.num_robots
                break
            inflated_grids.append(inflated_grid)
        return inflated_grids


class BaseMRLSPPlanner(MRPlanner):
    def __init__(self, robots, args, verbose=False):
        super(BaseMRLSPPlanner, self).__init__(robots, args)
        self.subgoals = set()
        self.selected_joint_action = None
        self.args = args
        self.verbose = verbose

        self.inflation_radius = args.inflation_radius_m / args.base_resolution
        if self.inflation_radius >= np.sqrt(5):
            self.downsample_factor = 2
        else:
            self.downsample_factor = 1

        self.update_counter = 0

    def update(self, observations, observed_maps, subgoals, robot_poses,
               visibility_masks):
        """Updates the internal state with the new grid/pose/laser scan.

        This function also computes a few necessary items, like which
        frontiers have recently been updated and computes their properties
        from the known grid.
        """
        self.update_counter += 1
        self.observations = observations
        # observed maps contain global occupancy grid for all robots and are same
        self.observed_maps = observed_maps
        self.robot_poses = robot_poses
        self.visibility_masks = visibility_masks
        # Since there's full communication, we can just use the first robot's observations as global
        self.observed_map = observed_maps[0]
        subgoals = subgoals[0]

        # Store the inflated grid after ensuring that the unreachable 'free
        # space' is set to 'unobserved'. This avoids trying to plan to
        # unreachable space and avoids having to check for this everywhere.
        self.inflated_grids = self._get_inflated_occupancy_grids()
        self.inflated_grid = mrlsp.utils.utility.get_fully_connected_global_grid_multirobot(
            self.inflated_grids[0], robot_poses)
        # Compute the new frontiers and update stored frontiers
        new_subgoals = set([copy.copy(s) for s in subgoals])
        self.subgoals = lsp.core.update_frontier_set(
            self.subgoals,
            new_subgoals,
            max_dist=2.0 / self.args.base_resolution)

        # Also check that the goal is not inside the frontier
        lsp.core.update_frontiers_goal_in_frontier(self.subgoals)

        # Update the subgoal inputs
        self._update_subgoal_inputs(observations['images'], robot_poses)

        # Once the subgoal inputs are set, compute their properties
        self._update_subgoal_properties(robot_poses)

    def _update_subgoal_inputs(self, images, robot_poses):
        # Loop through subgoals and get the 'input data'
        subgoal_distances = mrlsp.utils.utility.get_robot_subgoal_distances(self.inflated_grid, self.robot_poses)

        for subgoal in self.subgoals:
            if subgoal.props_set:
                continue
            # find subgoal that is close to the robot
            distance_to_subgoal = [subgoal_distances[i][subgoal] for i in range(self.num_robots)]
            robot_idx = np.argmin(distance_to_subgoal)
            # Compute the data that will be passed to the neural net
            input_data = lsp.utils.learning_vision.get_oriented_input_data(
                images[robot_idx], robot_poses[robot_idx], subgoal)

            # Store the input data alongside each subgoal
            subgoal.nn_input_data = input_data

    def _update_subgoal_properties(self, robot_pose):
        raise NotImplementedError("Method for abstract class")
