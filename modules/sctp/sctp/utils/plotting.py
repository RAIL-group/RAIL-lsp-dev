import numpy as np
import matplotlib.pyplot as plt
from sctp.param import RobotType
from scipy.stats import gaussian_kde

def plot_sctpgraph(graph, name="Testing Graph", path=None, 
               startID=None, goalID=None, seed=None):
    """Plot graph using matplotlib."""
    plt.figure(figsize=(10, 10))

    # Plot edges
    for edge in graph.edges:
        x_values = [edge.v1.coord[0], edge.v2.coord[0]]
        y_values = [edge.v1.coord[1], edge.v2.coord[1]]
        plt.plot(x_values, y_values, 'b-', alpha=0.7)
        # Display block probability
        mid_x = (edge.v1.coord[0] + edge.v2.coord[0]) / 2
        mid_y = (edge.v1.coord[1] + edge.v2.coord[1]) / 2
        costs = f"{edge.cost:.1f}"
        plt.text(mid_x, mid_y+0.25, costs, color='red', fontsize=8)

    # Plot nodes
    for node in graph.vertices:
        plt.scatter(node.coord[0], node.coord[1], color='blue', s=60)
        plt.text(node.coord[0], node.coord[1] + 0.2, f"V{node.id}", color='blue', fontsize=10)
        if startID is not None:
            if node.id == startID:
                plt.text(node.coord[0] - 0.2, node.coord[1] - 0.5, "S", color='blue', fontsize=15)
        if goalID is not None:
            if node.id == goalID:
                plt.text(node.coord[0] + 0.4, node.coord[1] - 0.4, "G", color='red', fontsize=15)

    for poi in graph.pois:
        plt.scatter(poi.coord[0], poi.coord[1], color='red', s=60)
        if poi.block_status == 1:
            plt.scatter(poi.coord[0], poi.coord[1], color='black', s=35)
        plt.text(poi.coord[0]-0.3, poi.coord[1] + 0.25, f"P{poi.id}"+f"/{poi.block_prob:.2f}", color='blue', fontsize=9)
        
    
    if path is not None:
        x_robot = []
        y_robot = []
        x_drone = []
        y_drone = []
        for a in path:
            if a.rtype == RobotType.Ground:
                x_robot.append(a.start_pose[0])
                y_robot.append(a.start_pose[1])
            elif a.rtype == RobotType.Drone:
                x_drone.append(a.start_pose[0])
                y_drone.append(a.start_pose[1])
        plt.plot(x_robot, y_robot, color='orange', linewidth=2)
        plt.scatter(x_robot, y_robot, color='orange',s=20)
        plt.plot(x_drone, y_drone, color='purple', linewidth=2)
    plt.title(name+f' | seed = {seed}')
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis("equal")
    plt.show()

def plot_sctpgraph_combine(graph, plt, verbose=False):

    # Plot edges
    for edge in graph.edges:
        x_values = [edge.v1.coord[0], edge.v2.coord[0]]
        y_values = [edge.v1.coord[1], edge.v2.coord[1]]
        plt.plot(x_values, y_values, 'b-', linewidth=1.0, alpha=1.0)
        # Display block probability
        if verbose:
            mid_x = (edge.v1.coord[0] + edge.v2.coord[0]) / 2
            mid_y = (edge.v1.coord[1] + edge.v2.coord[1]) / 2
            costs = f"{edge.cost:.1f}"
            plt.text(mid_x, mid_y+0.25, costs, color='red', fontsize=6)

    # Plot nodes
    for node in graph.vertices:
        plt.scatter(node.coord[0], node.coord[1], color='green', s=25)
        if verbose:
            plt.text(node.coord[0], node.coord[1] + 0.2, f"V{node.id}", color='blue', fontsize=6)

    for poi in graph.pois:
        plt.scatter(poi.coord[0], poi.coord[1], color='red', s=25)
        if poi.block_status == 1:
            plt.scatter(poi.coord[0], poi.coord[1], color='black', s=18)
        if verbose:
            plt.text(poi.coord[0]-0.3, poi.coord[1] + 0.25, f"POI{poi.id}"+f"/{poi.block_prob:.2f}", color='blue', fontsize=6)
        else:
            plt.text(poi.coord[0], poi.coord[1] + 0.1, f"P{poi.id}", color='blue', fontsize=6)
        

def make_scatter_plot_with_box(data_x, data_y, max_val=None, xlabel='Baseline', ylabel='SCTP'):
    if max_val is None:
        max_val = 1.1 * max(max(data_x), max(data_y))
    fig = plt.figure(figsize=(5, 5), dpi=300)
    gs = fig.add_gridspec(nrows=8, ncols=8)
    f_ax_mid = fig.add_subplot(gs[:, :])
    f_ax_bot = fig.add_subplot(gs[-1, :])
    f_ax_lft = fig.add_subplot(gs[:, 0])

    make_scatter_plot(f_ax_mid, data_x, data_y, max_val)

    f_ax_lft.boxplot(list(data_y), vert=True, showmeans=True)
    f_ax_lft.set_ylim([0, max_val])
    f_ax_lft.set_axis_off()

    f_ax_bot.boxplot(list(data_x), vert=False, showmeans=True)
    f_ax_bot.set_xlim([0, max_val])
    f_ax_bot.set_axis_off()

    f_ax_mid.set_xlabel(xlabel)
    f_ax_mid.set_ylabel(ylabel)

    baseline_cost = np.average(data_x)
    learned_cost = np.average(data_y)
    improv = (baseline_cost - learned_cost) / baseline_cost * 100

    f_ax_mid.set_title(f"{xlabel}: {baseline_cost:.1f}, {ylabel}: {learned_cost:.1f}, Improv: {improv:.1f} %")

    return f_ax_mid

def make_scatter_plot(ax, cost_x, cost_y, max_val):
    y_axis = cost_y
    x_axis = cost_x
    # Calculate the point density
    xy = np.vstack([x_axis, y_axis])
    z = gaussian_kde(xy)(xy)
    colors = plt.get_cmap("Blues")((z - z.min()) / (z.max() - z.min()) * 0.75 + 0.50)

    ax.scatter(x_axis, y_axis, c=colors)
    # Draw a center line
    ax.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=0.5, alpha=0.2)

    # Set X-axis and Y-axis up to same range; Using ax.axis('square') gives scaling issues for box plot
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)


def print_graph(nodes, edges, show_edge=False, show_node=False):
    """Print the graph details."""
    if show_node:
        for node in nodes:
            print(f"Node {node.id}: ({node.coord[0]:.2f}, {node.coord[1]:.2f}) with neighbors: {node.neighbors}")
            print(f"The neighbors features:")
            for n in node.neighbors:
                edge = [edge for edge in edges if ((edge.v1.id == node.id and edge.v2.id == n) or (edge.v1.id == n and edge.v2.id == node.id))][0]
                print(f"edge {edge.id}: block prob {edge.block_prob:.2f}, block status {edge.block_status}, cost {edge.cost}")
    if show_edge:
        for edge in edges:
            print(f"Edge {edge.id}: block prob {edge.block_prob:.2f}, block status {edge.block_status}, cost {edge.cost}")
