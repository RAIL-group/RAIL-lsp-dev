from object_search.core import Subgoal


class Planner():
    """Abstract class for all planners"""
    def __init__(self, target_obj_info, args, verbose=True):
        self.target_obj_info = target_obj_info
        self.args = args
        self.verbose = verbose

    def update(self, graph, grid, subgoals, robot_pose):
        self.graph = graph
        self.grid = grid
        self.robot_pose = robot_pose
        self.new_subgoals = [s for s in subgoals]
        self.subgoals = []
        for idx in subgoals:
            pose = self.graph.get_node_position_by_idx(idx)[:2]
            self.subgoals.append(Subgoal(idx, pose))
        self._update_subgoal_properties()

    def _update_subgoal_properties(self):
        pass

    def compute_selected_subgoal(self):
        raise NotImplementedError()


class BaseFrontierPlanner(Planner):

    def update(self, graph, grid, containers, frontiers, robot_pose):
        self.graph = graph
        self.grid = grid
        self.containers = []
        for idx in containers:
            pose = self.graph.get_node_position_by_idx(idx)[:2]
            self.containers.append(Subgoal(idx, pose))
        self.frontiers = [f for f in frontiers]
        self.robot_pose = robot_pose
        self.subgoals = self.containers + self.frontiers

        self._update_subgoal_properties()
