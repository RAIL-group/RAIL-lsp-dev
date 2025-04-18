import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.morphology import erosion
import networkx as nx
from .utils import get_object_color_from_type


COLLISION_VAL = 1
FREE_VAL = 0
UNOBSERVED_VAL = -1
assert (COLLISION_VAL > FREE_VAL)
assert (FREE_VAL > UNOBSERVED_VAL)
OBSTACLE_THRESHOLD = 0.5 * (FREE_VAL + COLLISION_VAL)


FOOT_PRINT = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])


def make_plotting_grid(grid_map):
    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * 0.75
    collision = grid_map >= OBSTACLE_THRESHOLD
    # Take one pixel boundary of the region collision
    thinned = erosion(collision, footprint=FOOT_PRINT)
    boundary = np.logical_xor(collision, thinned)
    free = np.logical_and(grid_map < OBSTACLE_THRESHOLD, grid_map >= FREE_VAL)
    grid[:, :, 0][free] = 1
    grid[:, :, 1][free] = 1
    grid[:, :, 2][free] = 1
    grid[:, :, 0][boundary] = 0
    grid[:, :, 1][boundary] = 0
    grid[:, :, 2][boundary] = 0

    return grid


def make_blank_grid(grid_map):
    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3])
    return grid


def plot_graph_on_grid(grid, graph):
    '''Plot the scene graph on the occupancy grid to scale'''
    plotting_grid = make_plotting_grid(np.transpose(grid))
    plt.imshow(plotting_grid)

    # find the room nodes
    room_node_idx = graph.room_indices

    rc_idx = room_node_idx + graph.container_indices

    # plot the edge connectivity between rooms and their containers only
    filtered_edges = [
        edge
        for edge in graph.edges
        if edge[1] in rc_idx and edge[0] != 0
    ]

    for (start, end) in filtered_edges:
        p1 = graph.nodes[start]['position']
        p2 = graph.nodes[end]['position']
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        plt.plot(x_values, y_values, 'c', linestyle="--", linewidth=0.3)

    # plot room nodes
    for room in rc_idx:
        room_pos = graph.nodes[room]['position']
        room_name = graph.nodes[room]['name']
        plt.text(room_pos[0], room_pos[1], room_name, color='brown',
                 size=6, rotation=40)


def plot_graph_on_grid_old(grid, graph):
    '''Plot the scene graph on the occupancy grid to scale'''
    plotting_grid = make_plotting_grid(np.transpose(grid))
    plt.imshow(plotting_grid)

    # find the room nodes
    room_node_idx = [idx for idx in range(1, graph['cnt_node_idx'][0])]

    rc_idx = room_node_idx + graph['cnt_node_idx']

    # plot the edge connectivity between rooms and their containers only
    filtered_edges = [
        edge
        for edge in graph['edge_index']
        if edge[1] in rc_idx and edge[0] != 0
    ]

    for (start, end) in filtered_edges:
        p1 = graph['nodes'][start]['pos']
        p2 = graph['nodes'][end]['pos']
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        plt.plot(x_values, y_values, 'c', linestyle="--", linewidth=0.3)

    # plot room nodes
    for room in rc_idx:
        room_pos = graph['nodes'][room]['pos']
        room_name = graph['nodes'][room]['name']
        plt.text(room_pos[0], room_pos[1], room_name, color='brown',
                 size=6, rotation=40)


def simulate_plan(trajectory, thor_data, args):
    print(f'Seed[{args.current_seed}]: Making video by simulating plan ...')
    fig = plt.figure()
    writer = animation.FFMpegWriter(12)
    writer.setup(fig, os.path.join(args.save_dir,
                 f'Eval_{args.current_seed}.mp4'), 500)

    fig_title = 'Seed: [' + str(args.current_seed) + \
                '] - Planner: [' + args.cost_str + '] - Step: '

    for step, grid_coord in enumerate(list(zip(trajectory[0], trajectory[1]))[::5]):
        position = thor_data.g2p_map[grid_coord]

        thor_data.controller.step(
            action="Teleport",
            position=position,
            horizon=30
        )
        plt.clf()
        top_down_frame = thor_data.get_top_down_frame()
        plt.imshow(top_down_frame)
        title = fig_title + str(step + 1)
        plt.title(title, fontsize='10')
        writer.grab_frame()
    writer.finish()
    print(f'Seed[{args.current_seed}]: Video Saved!')


def plot_plan(plan):
    textstr = ''
    for p in plan:
        textstr += str(p) + '\n'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place a text box in upper left in axes coords
    plt.text(0, 1, textstr, transform=plt.gca().transAxes, fontsize=5,
             verticalalignment='top', bbox=props)
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])

    # Add labels and title
    plt.title('Plan progression', fontsize=6)


def plot_graph(ax, nodes, edges, highlight_node=None):
    G = nx.Graph()
    node_colors = []
    for k, v in nodes.items():
        G.add_node(k, label=f"{k}: {v['name']}")
        color = get_object_color_from_type(v['type']) if k != highlight_node else 'cyan'
        node_colors.append(color)
    G.add_edges_from(edges)
    node_labels = nx.get_node_attributes(G, 'label')
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax, with_labels=True, labels=node_labels, node_color=node_colors, node_size=20,
            font_size=4, font_weight='regular', edge_color='black', width=0.5)
