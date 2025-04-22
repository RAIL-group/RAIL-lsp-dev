from lsp_select.planners import PolicySelectionPlanner
from mrlsp_select import offline_replay


class MRPolicySelectionPlanner(PolicySelectionPlanner):
    """Meta-planner class that handles selection among multiple planners/policies."""

    def update(self, observations, observed_maps, subgoals, robot_poses, visibility_masks):
        """Updates the information in currently chosen planner and records observed poses/images."""
        self.robot_poses = robot_poses
        self.observations = observations
        self.observed_map = observed_maps[0]
        self.planners[self.chosen_planner_idx].update(observations, observed_maps, subgoals, robot_poses, visibility_masks)
        self.inflated_grid = self.planners[self.chosen_planner_idx].inflated_grid
        self.subgoals = self.planners[self.chosen_planner_idx].subgoals

        poses = [[p.x, p.y, p.yaw] for p in robot_poses]
        self.poses.extend(poses)
        self.images.extend(observations['images'])

        # Update nearest pose data
        self.update_nearest_pose_data(visibility_masks, [p[:2] for p in poses])
        self.counter += len(self.robot_poses)

    def update_nearest_pose_data(self, visibility_masks, current_poses):
        """As the robot navigates, update pose_data to reflect the nearest robot pose
        from all poses in the visibility region.
        """
        for offset, (visibility_mask, current_pose) in enumerate(zip(visibility_masks, current_poses)):
            super(MRPolicySelectionPlanner, self).update_nearest_pose_data(visibility_mask, current_pose, offset)

    def _get_lb_costs(self, planner):
        """Get lower bound costs for a given planner."""
        optimistic_lb, simply_connected_lb = offline_replay.get_lowerbound_planner_costs(self.navigation_data,
                                                                                         planner,
                                                                                         self.args)
        return optimistic_lb, simply_connected_lb
