from taskplan_select.simulators import SceneGraphSimulator
from taskplan.planners import PlanningLoop
import taskplan


class OfflineReplay(SceneGraphSimulator):

    def __init__(self,
                 navigation_data,
                 partial_graph,
                 args,
                 target_obj_info,
                 partial_grid=None,
                 verbose=True):
        super().__init__(known_graph=partial_graph,
                         args=args,
                         target_obj_info=target_obj_info,
                         known_grid=partial_grid,
                         thor_interface=None,
                         verbose=verbose)
        self.navigation_data = navigation_data

    def get_top_down_image(self):
        return self.navigation_data['top_down_image']

    def initialize_graph_map_and_subgoals(self):
        return (self.navigation_data['graph'][0],
                self.known_grid,
                self.navigation_data['subgoals'][0])


def get_lowerbound_planner_costs(navigation_data, planner, args):
    partial_graph = navigation_data['graph'][-1]
    partial_grid = navigation_data['final_partial_grid']
    robot_pose = navigation_data['robot_pose']
    target_obj_info = navigation_data['target_obj_info']
    robot = taskplan.robot.Robot(robot_pose)
    simulator = OfflineReplay(navigation_data, partial_graph, args, target_obj_info, partial_grid)
    planning_loop = PlanningLoop(target_obj_info, simulator, robot, args, verbose=True)
    for counter, step_data in enumerate(planning_loop):
        planner.update(step_data['observed_graph'],
                       step_data['observed_grid'],
                       step_data['subgoals'],
                       step_data['robot_pose'])
        chosen_subgoal = planner.compute_selected_subgoal()
        planning_loop.set_chosen_subgoal(chosen_subgoal)

        if args.do_plot:
            pass
    net_motion, trajectory = taskplan.core.compute_path_cost(partial_grid, planning_loop.robot.all_poses)

    optimistic_lb = net_motion
    simply_connected_lb = net_motion
    return optimistic_lb, simply_connected_lb
