import lsp
import copy


class MRobot(lsp.robot.Turtlebot_Robot):
    def __init__(self, robot_index, num_robots, pose, primitive_length, num_primitives, map_data):
        super(MRobot, self).__init__(pose, primitive_length=primitive_length, num_primitives=num_primitives, map_data=map_data)
        # current robot's variables
        self.tag = robot_index
        self.timestamp = 0
        self.local_occupancy_grid = None
        self.global_occupancy_grid = None
        self.chosen_subgoal = None
        # Robot team variables
        self.team_subgoals = [None for i in range(num_robots)]
        self.team_poses = [None for i in range(num_robots)]
        self.team_all_poses = [[] for i in range(num_robots)]
        self.team_occupancy_grids = [None for i in range(num_robots)]

    def move(self, motion_primitives, index, data):
        super().move(motion_primitives, index)
        # All the updates that are only limited to the robot itself are done here.
        self.chosen_subgoal = data['subgoal']
        self.timestamp = data['timestamp']
        self._update_information()

    def _update_information(self):
        self.team_poses[self.tag] = self.pose
        self.team_all_poses[self.tag] = self.all_poses
        self.team_subgoals[self.tag] = self.chosen_subgoal
