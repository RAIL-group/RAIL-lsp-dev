import numpy as np
import matplotlib.pyplot as plt
from pddlstream.algorithms.search import solve_from_pddl

import procthor
import taskplan
from taskplan.environments.home import Home


def get_args():
    args = lambda: None
    args.current_seed = 0
    args.resolution = 0.05
    args.goal_type = 'breakfast+coffee'
    args.cost_type = 'learned'
    args.cache_path = '/data/.cache'
    args.save_dir = '/data/test_logs'
    args.network_file = '/data/sbert/logs/test/fcnn.pt'
    args.image_filename = f'demo_{args.current_seed}.png'
    args.logfile_name = 'task_learned_logfile.txt'
    return args


def test_home():
    args = get_args()

    # Load data for a given seed
    thor_data = Home()

    # Get the occupancy grid from data
    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()
    args.robot_room_coord = taskplan.utilities.utils.get_robots_room_coords(
        thor_data.occupancy_grid, init_robot_pose, thor_data.rooms)

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph, grid)
    partial_map.set_room_info(init_robot_pose, thor_data.rooms)

    if args.cost_type == 'learned':
        learned_data = {
            'partial_map': partial_map,
            'learned_net': args.network_file
        }
    else:
        learned_data = None
    # Instantiate PDDL for this map
    pddl = taskplan.pddl.helper.get_pddl_instance(
        whole_graph=whole_graph,
        map_data=thor_data,
        args=args,
        learned_data=learned_data,
        goal_for='demo'
    )
    taskplan.utilities.utils.check_pddl_validity(pddl, args)

    pddl['problem'] = taskplan.pddl.helper.\
        generate_pddl_problem_from_struct(pddl['problem_struct'])

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'],
                                 planner=pddl['planner'], max_planner_time=120)

    executed_actions, robot_poses, action_cost = taskplan.planners.task_loop.run(
        plan, pddl, partial_map, init_robot_pose, args)

    distance, trajectory = taskplan.core.compute_path_cost(partial_map.grid, robot_poses)
    distance += action_cost
    print(f"Planning cost: {distance}")

    # plotting code
    plt.clf()
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"{pddl['goal']} - seed: [{args.current_seed}]", fontsize=6)

    plt.subplot(221)
    # 0 plot the plan
    taskplan.plotting.plot_plan(plan=executed_actions)

    plt.subplot(222)
    # 1 plot the whole graph
    plt.title('Whole scene graph', fontsize=6)
    graph_image = whole_graph['graph_image']
    plt.imshow(graph_image)
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])

    viridis_cmap = plt.get_cmap('viridis')

    colors = np.linspace(0, 1, len(trajectory[0]))
    line_colors = viridis_cmap(colors)

    plt.subplot(223)
    # 3 plot the graph overlaied image
    procthor.plotting.plot_graph_on_grid_old(grid, whole_graph)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title('Graph overlaied occupancy grid', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.subplot(224)
    # 4 plot the graph overlaied image
    plotting_grid = procthor.plotting.make_plotting_grid(np.transpose(grid))
    plt.imshow(plotting_grid)
    x, y = init_robot_pose
    plt.text(x, y, '+', color='red', size=6, rotation=45)
    plt.title(f'Cost: {distance:0.3f}', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        plt.plot(x, y, color=line_colors[idx], marker='.', markersize=3)

    plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=1000)
