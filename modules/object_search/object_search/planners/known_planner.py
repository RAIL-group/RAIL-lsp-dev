from .planner import Planner
from lsp.core import get_robot_distances


class KnownPlanner(Planner):
    '''This planner uses known graph to find the target object in one step.'''
    def __init__(self, target_obj_info, args, known_graph, known_grid, destination=None, verbose=True):
        super(KnownPlanner, self).__init__(
            target_obj_info, args, verbose)
        self.known_graph = known_graph
        self.known_grid = known_grid
        self.destination = destination

    def _update_subgoal_properties(self):
        for subgoal in self.subgoals:
            contained_obj_idx = self.known_graph.get_adjacent_nodes_idx(subgoal.id, filter_by_type=3)
            contained_obj_names = [self.known_graph.get_node_name_by_idx(idx) for idx in contained_obj_idx]
            if self.target_obj_info['name'] in contained_obj_names:
                subgoal.set_props(prob_feasible=1.0)
            else:
                subgoal.set_props(prob_feasible=0.0)
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:.2f} | '
                    f'for {self.known_graph.get_node_name_by_idx(subgoal.id)}'
                )
        if self.verbose:
            print(f'Containers with target object [{self.target_obj_info["name"]}]:')
            for subgoal in self.subgoals:
                print([f'{subgoal.id}: {self.known_graph.get_node_name_by_idx(subgoal.id)}'
                       for subgoal in self.subgoals if subgoal.prob_feasible == 1.0])
            print(" ")

    def compute_selected_subgoal(self):
        feasible_subgoals = [subgoal for subgoal in self.subgoals if subgoal.prob_feasible == 1.0]
        robot_distances = get_robot_distances(
            self.known_grid, self.robot_pose, feasible_subgoals)
        min_cost_subgoal = min(feasible_subgoals, key=robot_distances.get)
        return min_cost_subgoal.id
