import potlp
import gridmap
from . import planner

class KnownPlanner(planner.BasePOTLPPlanner):
    def __init__(self, args, known_map, ltl_planner, all_nodes, robot, simulator, verbose=False, iterations=10000):
        super(KnownPlanner, self).__init__(args, known_map, ltl_planner, all_nodes, robot, simulator, verbose=verbose, iterations=iterations)
        inflation_radius = args.inflation_radius_m / args.base_resolution
        self.inflated_known_grid = gridmap.utils.inflate_grid(
            self.known_map, inflation_radius=inflation_radius)
    
    
    def get_subgoal_props(self, subgoal_nodes):
        subgoal_prop_dict = potlp.core.get_known_subgoal_props_dict_updated(self.inflated_known_grid,
                                self.inflated_grid, self.known_space_nodes_ordered, subgoal_nodes, self.robot)
        return subgoal_prop_dict
