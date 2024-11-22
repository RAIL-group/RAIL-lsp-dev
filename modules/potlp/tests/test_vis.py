import potlp
import environments
import pytest
import matplotlib.pyplot as plt
import lsp
import mrlsp

def get_args():
    args = lambda: None
    args.current_seed = 2006
    args.map_type = 'maze'
    args.unity_path = '/unity/rail_sim.x86_64'
    args.save_dir = './'

    # Robot Arguments
    args.step_size = 1.8
    args.num_primitives = 32
    args.laser_scanner_num_points = 1024
    args.field_of_view_deg = 360

    # For the simulator
    args.disable_known_grid_correction = False
    return args

def test_vis_potlp_known_planner(do_plot=True):
    args = get_args()
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    # Specification
    specification = "(!start U goal1) & (!start U goal2) & F start"
    # specification = "F goal1 | F goal2"
    # specification = "(!goal1 U goal2) & F goal1"
    ltl_planner = potlp.core.LTLPlanner(specification, only_singular_transitions=True)

    res = args.base_resolution
    start_node = potlp.core.Node(props=("start",), is_subgoal=False)
    start_node.position = (int(pose.x), int(pose.y))
    goal1_node = potlp.core.Node(props=("goal1",), is_subgoal=False)
    goal1_node.position = (int(24.6/res), int(12/res))
    goal2_node = potlp.core.Node(props=("goal2",), is_subgoal=False)
    goal2_node.position = (int(5.2/res), int(39/res))
    all_nodes = [
        goal1_node, goal2_node, start_node
    ]

    robot = lsp.robot.Turtlebot_Robot(
                            pose,
                            primitive_length=args.step_size,
                            num_primitives=args.num_primitives,
                            map_data=map_data)
    
    simulator = lsp.simulators.Simulator(known_map, goal, args)
    known_planner = potlp.planners.KnownPlanner(args,
                        known_map,
                        ltl_planner,
                        all_nodes,
                        robot,
                        simulator,
                        verbose=True,
                        iterations=1000
                    )

    while not ltl_planner.is_dfa_state_accepting(known_planner.dfa_state):
        known_planner.step()
        print(ltl_planner.semantic_index)
        # plotting
        if do_plot:
            plt.ion()
            plt.figure(1, figsize=(12, 4))
            plt.clf()
            ax = plt.gca()
            full_path = potlp.core.get_path_from_node_path(known_planner.planning_grid, robot.pose, known_planner.node_path)
            mrlsp.utils.plotting.plot_grid_with_frontiers(
                    ax, known_planner.robot_grid, known_map, known_planner.frontiers)
            mrlsp.utils.plotting.plot_pose_path(ax, robot.all_poses)
            ax.plot(full_path[0, :], full_path[1, :], 'b:')
            ax.scatter([robot.pose.x], [robot.pose.y], c='r', s=10)
            for ks_node in all_nodes:
                plt.plot([ks_node.position[0]],
                            [ks_node.position[1]], '.')
                plt.text(ks_node.position[0], ks_node.position[1], ks_node.props[0])
            plt.pause(0.1)
            plt.show()
