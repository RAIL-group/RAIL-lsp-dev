import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
import numpy as np
import lsp

colors = plt.get_cmap('tab10').colors


def plot_pose(ax, pose, color='black', filled=True):
    if filled:
        ax.scatter(pose.x, pose.y, color=color, s=10, label='point')
    else:
        ax.scatter(pose.x,
                   pose.y,
                   color=color,
                   s=10,
                   label='point',
                   facecolors='none')


def plot_path(ax, path, color='blue', linestyle=':'):
    if path is not None:
        ax.plot(path[0, :], path[1, :], color=color, linestyle=linestyle)


def plot_grid_with_frontiers(ax,
                             grid_map,
                             known_map,
                             frontiers,
                             cmap_name='viridis'):
    grid = lsp.utils.plotting.make_plotting_grid(grid_map, known_map)

    cmap = plt.get_cmap(cmap_name)
    for frontier in frontiers:
        color = cmap(frontier.prob_feasible)
        grid[frontier.points[0, :], frontier.points[1, :], 0] = color[0]
        grid[frontier.points[0, :], frontier.points[1, :], 1] = color[1]
        grid[frontier.points[0, :], frontier.points[1, :], 2] = color[2]

    ax.imshow(np.transpose(grid, (1, 0, 2)))


def plot_pose_path(ax, poses, color='blue', linestyle='-'):
    path = np.array([[p.x, p.y] for p in poses]).T
    plot_path(ax, path, color=color, linestyle=linestyle)


def highlight_subgoal(ax, subgoal, color='red', s=5):
    frontier_points = subgoal.points
    ax.scatter(frontier_points[0, :], frontier_points[1, :], color=color, s=s)


def visualize_robots(args, robots, observed_map, pano_images, subgoals, goal_pose, paths, known_map, timestamp):
    for i in range(args.num_robots):
        # Top axes: plot panoramic image
        ax = plt.subplot(args.num_robots, 2, 2*i+1)
        plt.imshow(pano_images[i])

    ax = plt.subplot(1, 2, 2)
    plot_grid_with_frontiers(ax, observed_map, known_map, subgoals)
    plot_pose(ax, goal_pose, filled=False)

    for i in range(args.num_robots):
        plot_pose(ax, robots[i].pose, color=colors[i])
        plot_pose_path(ax, robots[i].all_poses, color=colors[i])
        if paths[i] is not None:
            plot_path(ax, paths[i], color=colors[i])


def visualize_mrlsp_result(ax, args, robots, observed_map, subgoals, goal_pose, paths, known_map):
    plot_grid_with_frontiers(ax, observed_map, known_map, subgoals)
    plot_pose(ax, goal_pose, filled=False)

    for i in range(args.num_robots):
        plot_pose(ax, robots[i].pose, color=colors[i])
        plot_pose_path(ax, robots[i].all_poses, color=colors[i])
        if paths[i] is not None:
            plot_path(ax, paths[i], color=colors[i])


def make_scatter_plot(ax, cost_x, cost_y, max_val, xlabel='Learned', ylabel='Baseline'):
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


def make_scatter_plot_with_box(data_x, data_y, max_val=None, xlabel='Baseline', ylabel='Learned'):
    if max_val is None:
        max_val = 1.1 * max(max(data_x), max(data_y))
    fig = plt.figure(figsize=(5, 5), dpi=300)
    gs = fig.add_gridspec(nrows=8, ncols=8)
    f_ax_mid = fig.add_subplot(gs[:, :])
    f_ax_bot = fig.add_subplot(gs[-1, :])
    f_ax_lft = fig.add_subplot(gs[:, 0])

    make_scatter_plot(f_ax_mid, data_x, data_y, max_val, xlabel, ylabel)

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

    f_ax_mid.set_title(f"{xlabel}: {baseline_cost:.1f}, {ylabel}: {learned_cost:.1f}, Improv: {improv:.1f}")

    return f_ax_mid
