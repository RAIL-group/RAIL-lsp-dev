from .planner import Planner


class OptimisticPlanner(Planner):
    '''This planner optimistically explores the nearest container to search the target object.'''
    def __init__(self, target_obj_info, args, destination=None, verbose=True):
        super(OptimisticPlanner, self).__init__(target_obj_info, args, verbose)
        self.destination = destination

    def compute_selected_subgoal(self):
        robot_distances = self.get_robot_distances(
            self.grid, self.robot_pose, self.subgoals)
        return min(self.subgoals, key=robot_distances.get)
